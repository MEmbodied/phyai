#include "radix_cache/prefix_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace phyai_ext::radix_cache {

namespace {

inline std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

prefix_cache::prefix_cache(std::uint32_t atom_bytes, std::uint32_t atoms_per_unit, std::array<tier_config, max_tiers> tiers,
                           std::string eviction_policy_name)
    : prefix_cache(prefix_cache_config{atom_bytes, atoms_per_unit, tiers, std::move(eviction_policy_name), /*slru=*/2,
                                       /*max_events=*/16384}) {}

prefix_cache::prefix_cache(prefix_cache_config cfg)
    : cfg_(std::move(cfg)),
      atom_bytes_(cfg_.atom_bytes),
      atoms_per_unit_(cfg_.atoms_per_unit),
      tiers_(cfg_.tiers),
      tree_(cfg_.atom_bytes, cfg_.atoms_per_unit),
      policy_(make_eviction_policy(cfg_.eviction_policy, cfg_.slru_threshold)) {
  if (atom_bytes_ == 0 || atoms_per_unit_ == 0) {
    throw cache_usage_error("prefix_cache: atom_bytes / atoms_per_unit must be > 0");
  }
  allocators_.reserve(max_tiers);
  for (std::size_t t = 0; t < max_tiers; ++t) {
    if (tiers_[t].enabled && tiers_[t].total_units > 0) {
      allocators_.push_back(std::make_unique<unit_allocator>(tier_from_index(t), tiers_[t].total_units));
    } else {
      allocators_.push_back(std::make_unique<unit_allocator>(tier_from_index(t), 0));
      tiers_[t].enabled = false;
    }
    pending_units_per_tier_[t] = 0;
  }
}

prefix_cache::match_result prefix_cache::match(atom_span query) {
  if (query.size() % atom_bytes_ != 0) {
    throw cache_usage_error("prefix_cache::match: query bytes not aligned to atom_bytes");
  }
  if (query.size() % (static_cast<std::size_t>(atom_bytes_) * atoms_per_unit_) != 0) {
    throw cache_usage_error("prefix_cache::match: query atom count must be multiple of atoms_per_unit");
  }
  auto walk = tree_.walk(query, clock::now());
  match_result out;
  for (std::size_t t = 0; t < max_tiers; ++t) {
    out.last_node[t] = walk.last_node[t];
    out.matched_atoms[t] = walk.matched_atoms[t];
  }
  return out;
}

std::size_t prefix_cache::collect_units_into(tree_node* end, tier t, std::int32_t* out, std::size_t capacity) const {
  if (end == nullptr || end == tree_.root()) return 0;
  // First pass: count total ids along the chain.
  std::size_t total = 0;
  for (tree_node* n = end; n != nullptr; n = n->parent()) {
    if (auto* r = n->resource(t)) total += r->size_in_units();
  }
  if (total > capacity || out == nullptr) return total;
  // Second pass: walk root -> end (reverse the parent chain) and copy in order.
  std::vector<tree_node*> chain;
  chain.reserve(16);
  for (tree_node* n = end; n != nullptr; n = n->parent()) {
    if (n->resource(t) != nullptr) chain.push_back(n);
  }
  std::size_t off = 0;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    auto ids = (*it)->resource(t)->ids();
    std::memcpy(out + off, ids.data(), ids.size() * sizeof(unit_id));
    off += ids.size();
  }
  return total;
}

namespace {
// CPU malloc allocator for tvm::ffi::Tensor. Owns the buffer so the tensor's
// DLPack deleter frees it.
struct cpu_i32_alloc {
  void AllocData(DLTensor* t) {
    std::size_t bytes = ::tvm::ffi::GetDataSize(*t);
    t->data = bytes == 0 ? nullptr : std::malloc(bytes);
  }
  void FreeData(DLTensor* t) {
    if (t->data != nullptr) {
      std::free(t->data);
      t->data = nullptr;
    }
  }
};
}  // namespace

tvm::ffi::Tensor prefix_cache::collect_units_dlpack(tree_node* end, tier t) const {
  // First pass to size the tensor.
  std::size_t n = 0;
  if (end != nullptr && end != tree_.root()) {
    for (tree_node* p = end; p != nullptr; p = p->parent()) {
      if (auto* r = p->resource(t)) n += r->size_in_units();
    }
  }
  std::int64_t shape[1] = {static_cast<std::int64_t>(n)};
  DLDataType dtype{kDLInt, 32, 1};
  DLDevice cpu{kDLCPU, 0};
  auto tensor = tvm::ffi::Tensor::FromNDAlloc(cpu_i32_alloc{}, tvm::ffi::ShapeView(shape, 1), dtype, cpu);
  if (n > 0) {
    auto* dst = static_cast<std::int32_t*>(tensor.data_ptr());
    [[maybe_unused]] std::size_t written = collect_units_into(end, t, dst, n);
    // `written` should equal `n`; assert in debug builds.
  }
  return tensor;
}

prefix_cache::insert_result prefix_cache::insert(tier t, atom_span atoms, owned_units&& units) {
  if (units.get_tier() != t) { throw cache_invariant_error("prefix_cache::insert: owned_units tier mismatch"); }
  if (atoms.size() % (static_cast<std::size_t>(atom_bytes_) * atoms_per_unit_) != 0) {
    throw cache_usage_error("prefix_cache::insert: atoms not page-aligned");
  }
  const std::size_t n_pages = atoms.size() / atom_bytes_ / atoms_per_unit_;
  if (units.size() != n_pages) { throw cache_usage_error("prefix_cache::insert: units count != #pages in atoms"); }

  auto walk = tree_.walk(atoms, clock::now());
  tree_node* base = walk.last_node[tier_index(t)];
  std::uint32_t matched = walk.matched_atoms[tier_index(t)];
  if (base == nullptr) {
    base = tree_.root();
    matched = 0;
  }

  const std::size_t overlap_units = matched / atoms_per_unit_;
  owned_units overlap = units.take_first(overlap_units);
  const auto freed = static_cast<std::uint32_t>(overlap.size());
  (void)overlap;

  atom_span suffix = atoms.subspan(static_cast<std::size_t>(matched) * atom_bytes_);
  tree_node* tail = (suffix.empty()) ? base : tree_.insert_suffix(base, suffix);

  std::vector<tree_node*> chain;
  for (tree_node* n = tail; n != base; n = n->parent()) chain.push_back(n);
  std::reverse(chain.begin(), chain.end());
  for (tree_node* n : chain) {
    if (n->has_resource(t)) continue;
    auto piece = units.take_first(n->atom_count() / atoms_per_unit_);
    n->attach_resource(t, std::make_unique<node_resource>(std::move(piece)));
    n->touch(clock::now());
    n->set_last_access_step(current_step_);
  }

  emit(cache_event{cache_event_kind::insert, t, t, null_op_handle, static_cast<std::uint32_t>(suffix.size() / atom_bytes_),
                   static_cast<std::uint32_t>(suffix.size() / atom_bytes_ / atoms_per_unit_), now_ns()});
  return {tail, static_cast<std::uint32_t>(suffix.size() / atom_bytes_), freed};
}

void prefix_cache::collect_evict_candidates(tier t, std::vector<tree_node*>& out) const {
  // DFS; collect every per-tier leaf — i.e. a node that has a resource for `t`
  // but no descendant in the same tier does. Iterative post-order, no
  // recursion (radix tree depth can reach hundreds).
  struct frame {
    const tree_node* n;
    bool processed;
  };
  std::vector<frame> stk;
  stk.push_back({tree_.root(), false});
  std::vector<const tree_node*> covered;
  covered.reserve(64);
  while (!stk.empty()) {
    auto& f = stk.back();
    if (!f.processed) {
      f.processed = true;
      for (auto& [_, c] : const_cast<tree_node*>(f.n)->children()) { stk.push_back({c.get(), false}); }
    } else {
      const tree_node* n = f.n;
      stk.pop_back();
      bool desc_has = false;
      for (auto& [_, c] : const_cast<tree_node*>(n)->children()) {
        if (c->has_resource(t)) {
          desc_has = true;
          break;
        }
        if (std::find(covered.begin(), covered.end(), c.get()) != covered.end()) {
          desc_has = true;
          break;
        }
      }
      if (desc_has) covered.push_back(n);
      if (n != tree_.root() && n->has_resource(t) && !desc_has) { out.push_back(const_cast<tree_node*>(n)); }
    }
  }
}

void prefix_cache::check_pending_budget(tier dst_tier, std::size_t need_units) const {
  const auto cap = tiers_[tier_index(dst_tier)].max_pending_units;
  if (cap <= 0) return;
  if (pending_units_per_tier_[tier_index(dst_tier)] + static_cast<std::int32_t>(need_units) > cap) {
    throw cache_capacity_error("prefix_cache: pending units cap reached on destination tier");
  }
}

std::uint32_t prefix_cache::evict_node_resource(tree_node* n, tier t, std::optional<tier> promote_to) {
  auto* res = n->resource(t);
  if (res == nullptr || !res->is_evictable()) return 0;
  auto units = n->detach_resource(t)->take_units();
  const auto freed = static_cast<std::uint32_t>(units.size());

  if (promote_to.has_value() && tiers_[tier_index(*promote_to)].enabled) {
    auto& dst_alloc = *allocators_[tier_index(*promote_to)];
    if (dst_alloc.available_units() >= static_cast<std::int32_t>(freed)) {
      try {
        check_pending_budget(*promote_to, freed);
      } catch (...) {
        // Pending budget exhausted on destination — fall through to drop.
        emit(cache_event{cache_event_kind::evict, t, t, null_op_handle, n->atom_count(), freed, now_ns()});
        for (auto& obs : observers_) obs.fn(n, t);
        if (n->is_orphan()) tree_.prune_empty(n);
        return freed;
      }
      auto dst_units = dst_alloc.allocate(freed);
      auto dst_res = std::make_unique<node_resource>(std::move(dst_units));
      op_handle h = next_handle();
      dst_res->set_pending(h);
      n->attach_resource(*promote_to, std::move(dst_res));
      {
        std::lock_guard<std::mutex> g(op_state_mtx_);
        inflight_.emplace(h, pending_op{n, t, *promote_to, /*is_promote=*/false, freed});
        pending_units_per_tier_[tier_index(*promote_to)] += static_cast<std::int32_t>(freed);
      }
      emit(cache_event{cache_event_kind::demote_start, t, *promote_to, h, n->atom_count(), freed, now_ns()});
      for (auto& obs : observers_) obs.fn(n, t);
      // Note: we do not prune the node here — its destination resource is
      // still attached as Pending.
      return freed;
    }
    // Destination has no room — fall through to drop.
  }

  emit(cache_event{cache_event_kind::evict, t, t, null_op_handle, n->atom_count(), freed, now_ns()});
  for (auto& obs : observers_) obs.fn(n, t);
  if (n->is_orphan()) tree_.prune_empty(n);
  return freed;
}

void prefix_cache::ensure_capacity(tier t, std::int32_t need_units, std::optional<tier> promote_to) {
  if (!tiers_[tier_index(t)].enabled) { throw cache_usage_error("ensure_capacity: tier not enabled"); }
  // Cascade: if we want to demote into `promote_to`, but `promote_to` is full,
  // first shrink `promote_to` itself.
  if (promote_to.has_value() && tiers_[tier_index(*promote_to)].enabled) {
    auto& dst_alloc = *allocators_[tier_index(*promote_to)];
    if (dst_alloc.available_units() < need_units) {
      // Best-effort cascade — drop oldest in dst tier without further chaining.
      std::vector<tree_node*> dst_cands;
      collect_evict_candidates(*promote_to, dst_cands);
      std::sort(dst_cands.begin(), dst_cands.end(), [&](tree_node* a, tree_node* b) {
        return policy_->priority(*a, *promote_to) < policy_->priority(*b, *promote_to);
      });
      std::int32_t dst_deficit = need_units - dst_alloc.available_units();
      for (tree_node* n : dst_cands) {
        if (dst_deficit <= 0) break;
        auto* r = n->resource(*promote_to);
        if (r == nullptr || !r->is_evictable()) continue;
        auto freed = evict_node_resource(n, *promote_to, std::nullopt);
        dst_deficit -= static_cast<std::int32_t>(freed);
      }
    }
  }

  auto& alloc = *allocators_[tier_index(t)];
  std::int32_t deficit = need_units - alloc.available_units();
  if (deficit <= 0) return;

  std::vector<tree_node*> cands;
  collect_evict_candidates(t, cands);
  std::sort(cands.begin(), cands.end(),
            [&](tree_node* a, tree_node* b) { return policy_->priority(*a, t) < policy_->priority(*b, t); });

  for (tree_node* n : cands) {
    if (deficit <= 0) break;
    auto* res = n->resource(t);
    if (res == nullptr || !res->is_evictable()) continue;
    auto freed = evict_node_resource(n, t, promote_to);
    deficit -= static_cast<std::int32_t>(freed);
  }
}

std::uint32_t prefix_cache::evict_by_predicate(tier t, const std::function<bool(const tree_node&)>& should_evict) {
  if (!tiers_[tier_index(t)].enabled) return 0;
  std::vector<tree_node*> cands;
  collect_evict_candidates(t, cands);
  std::uint32_t total = 0;
  for (tree_node* n : cands) {
    auto* res = n->resource(t);
    if (res == nullptr || !res->is_evictable()) continue;
    if (!should_evict(*n)) continue;
    total += evict_node_resource(n, t, std::nullopt);
  }
  return total;
}

std::uint32_t prefix_cache::evict_by_named_predicate(tier t, const std::string& predicate, std::int64_t arg) {
  if (predicate == "step_le") { return evict_by_predicate(t, predicate_step_le(static_cast<std::uint64_t>(arg))); }
  if (predicate == "step_lt") { return evict_by_predicate(t, predicate_step_lt(static_cast<std::uint64_t>(arg))); }
  if (predicate == "hits_lt") { return evict_by_predicate(t, predicate_hits_lt(static_cast<std::uint64_t>(arg))); }
  if (predicate == "priority_le") { return evict_by_predicate(t, predicate_priority_le(arg)); }
  if (predicate == "age_ns_ge") { return evict_by_predicate(t, predicate_age_ns_ge(arg, clock::now())); }
  throw cache_usage_error("evict_by_named_predicate: unknown predicate '" + predicate + "'");
}

owned_units prefix_cache::allocate(tier t, std::int32_t n) {
  if (!tiers_[tier_index(t)].enabled) { throw cache_usage_error("allocate: tier not enabled"); }
  return allocators_[tier_index(t)]->allocate(static_cast<std::size_t>(n));
}

std::int32_t prefix_cache::available(tier t) const { return allocators_[tier_index(t)]->available_units(); }

std::int32_t prefix_cache::total(tier t) const { return allocators_[tier_index(t)]->total_units(); }

std::int32_t prefix_cache::active(tier t) const { return allocators_[tier_index(t)]->active_units(); }

std::int32_t prefix_cache::pending_units(tier t) const {
  std::lock_guard<std::mutex> g(op_state_mtx_);
  return pending_units_per_tier_[tier_index(t)];
}

op_handle prefix_cache::start_demote(tree_node* node, tier src_tier, tier dst_tier) {
  if (!tiers_[tier_index(src_tier)].enabled || !tiers_[tier_index(dst_tier)].enabled) {
    throw cache_usage_error("start_demote: src/dst tier disabled");
  }
  auto* src_res = node->resource(src_tier);
  if (src_res == nullptr) { throw cache_invariant_error("start_demote: node has no src tier resource"); }
  const auto need = src_res->size_in_units();
  check_pending_budget(dst_tier, need);
  auto dst_units = allocators_[tier_index(dst_tier)]->allocate(need);
  auto dst_res = std::make_unique<node_resource>(std::move(dst_units));
  op_handle h = next_handle();
  dst_res->set_pending(h);
  node->attach_resource(dst_tier, std::move(dst_res));
  {
    std::lock_guard<std::mutex> g(op_state_mtx_);
    inflight_.emplace(h, pending_op{node, src_tier, dst_tier, /*is_promote=*/false, static_cast<std::uint32_t>(need)});
    pending_units_per_tier_[tier_index(dst_tier)] += static_cast<std::int32_t>(need);
  }
  emit(cache_event{cache_event_kind::demote_start, src_tier, dst_tier, h, node->atom_count(), static_cast<std::uint32_t>(need),
                   now_ns()});
  return h;
}

op_handle prefix_cache::start_promote(tree_node* node, tier src_tier, tier dst_tier) {
  if (!tiers_[tier_index(src_tier)].enabled || !tiers_[tier_index(dst_tier)].enabled) {
    throw cache_usage_error("start_promote: src/dst tier disabled");
  }
  auto* src_res = node->resource(src_tier);
  if (src_res == nullptr) { throw cache_invariant_error("start_promote: node has no src tier resource"); }
  const auto need = src_res->size_in_units();
  check_pending_budget(dst_tier, need);
  auto dst_units = allocators_[tier_index(dst_tier)]->allocate(need);
  auto dst_res = std::make_unique<node_resource>(std::move(dst_units));
  op_handle h = next_handle();
  dst_res->set_pending(h);
  node->attach_resource(dst_tier, std::move(dst_res));
  {
    std::lock_guard<std::mutex> g(op_state_mtx_);
    inflight_.emplace(h, pending_op{node, src_tier, dst_tier, /*is_promote=*/true, static_cast<std::uint32_t>(need)});
    pending_units_per_tier_[tier_index(dst_tier)] += static_cast<std::int32_t>(need);
  }
  emit(cache_event{cache_event_kind::promote_start, src_tier, dst_tier, h, node->atom_count(), static_cast<std::uint32_t>(need),
                   now_ns()});
  return h;
}

void prefix_cache::complete_op(op_handle handle, bool success) {
  pending_op p{};
  {
    std::lock_guard<std::mutex> g(op_state_mtx_);
    auto it = inflight_.find(handle);
    if (it == inflight_.end()) { throw cache_usage_error("complete_op: unknown handle"); }
    p = it->second;
    inflight_.erase(it);
    pending_units_per_tier_[tier_index(p.dst_tier)] -= static_cast<std::int32_t>(p.unit_count);
  }
  op_done_cv_.notify_all();

  auto* dst_res = p.node->resource(p.dst_tier);
  if (dst_res == nullptr) { return; }
  if (!success) {
    p.node->detach_resource(p.dst_tier);
    if (p.node->is_orphan()) tree_.prune_empty(p.node);
    emit(cache_event{p.is_promote ? cache_event_kind::promote_fail : cache_event_kind::demote_fail, p.src_tier, p.dst_tier,
                     handle, p.node->atom_count(), 0, now_ns()});
    return;
  }
  dst_res->set_ready();
  if (!p.is_promote) {
    auto src = p.node->detach_resource(p.src_tier);
    (void)src;
    emit(cache_event{cache_event_kind::demote_done, p.src_tier, p.dst_tier, handle, p.node->atom_count(),
                     static_cast<std::uint32_t>(dst_res->size_in_units()), now_ns()});
  } else {
    emit(cache_event{cache_event_kind::promote_done, p.src_tier, p.dst_tier, handle, p.node->atom_count(),
                     static_cast<std::uint32_t>(dst_res->size_in_units()), now_ns()});
  }
}

std::vector<op_handle> prefix_cache::inflight_ops() const {
  std::lock_guard<std::mutex> g(op_state_mtx_);
  std::vector<op_handle> out;
  out.reserve(inflight_.size());
  for (auto& [h, _] : inflight_) out.push_back(h);
  return out;
}

bool prefix_cache::wait_op(op_handle handle, std::int64_t timeout_ms) {
  std::unique_lock<std::mutex> g(op_state_mtx_);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<std::int64_t>(timeout_ms, 0));
  while (inflight_.find(handle) != inflight_.end()) {
    if (timeout_ms <= 0) return inflight_.find(handle) == inflight_.end();
    if (op_done_cv_.wait_until(g, deadline) == std::cv_status::timeout) { return inflight_.find(handle) == inflight_.end(); }
  }
  return true;
}

std::uint32_t prefix_cache::fail_all_inflight() {
  std::vector<op_handle> handles;
  {
    std::lock_guard<std::mutex> g(op_state_mtx_);
    handles.reserve(inflight_.size());
    for (auto& [h, _] : inflight_) handles.push_back(h);
  }
  std::uint32_t count = 0;
  for (auto h : handles) {
    try {
      complete_op(h, /*success=*/false);
      ++count;
    } catch (const std::exception&) {
      // op may have been completed concurrently; ignore.
    }
  }
  return count;
}

std::size_t prefix_cache::add_evict_observer(evict_observer obs) {
  observers_.push_back({next_observer_id_, std::move(obs)});
  return next_observer_id_++;
}

bool prefix_cache::remove_evict_observer(std::size_t observer_id) {
  auto it = std::find_if(observers_.begin(), observers_.end(),
                         [observer_id](const observer_entry& e) { return e.id == observer_id; });
  if (it == observers_.end()) return false;
  observers_.erase(it);
  return true;
}

std::vector<cache_event> prefix_cache::take_events() {
  std::lock_guard<std::mutex> g(events_mtx_);
  std::vector<cache_event> out;
  out.swap(events_);
  return out;
}

std::size_t prefix_cache::dropped_events() const {
  std::lock_guard<std::mutex> g(events_mtx_);
  return dropped_events_;
}

void prefix_cache::touch_step(tree_node* node) {
  if (node == nullptr) return;
  for (tree_node* n = node; n != nullptr; n = n->parent()) {
    n->set_last_access_step(current_step_);
    n->touch(clock::now());
  }
}

std::uint64_t prefix_cache::node_path_hash(const tree_node* node) const {
  if (node == nullptr || node == tree_.root()) return 0;
  // Walk root -> node, hashing each segment. Using xxh3 on the concatenated
  // bytes is equivalent to a single hash over the full path.
  std::vector<const tree_node*> chain;
  for (const tree_node* n = node; n != nullptr && n != tree_.root(); n = n->parent()) { chain.push_back(n); }
  std::reverse(chain.begin(), chain.end());
  atom_vec buf;
  for (const tree_node* n : chain) {
    auto a = n->atoms();
    buf.insert(buf.end(), a.begin(), a.end());
  }
  return xxh3_64(buf.data(), buf.size());
}

void prefix_cache::emit(cache_event ev) {
  std::lock_guard<std::mutex> g(events_mtx_);
  if (cfg_.max_events_buffered > 0 && events_.size() >= cfg_.max_events_buffered) {
    events_.erase(events_.begin());
    ++dropped_events_;
  }
  events_.push_back(ev);
}

op_handle prefix_cache::next_handle() {
  std::lock_guard<std::mutex> g(op_state_mtx_);
  return next_handle_++;
}

}  // namespace phyai_ext::radix_cache
