#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "radix_cache/atoms.h"
#include "radix_cache/events.h"
#include "radix_cache/owned_units.h"
#include "radix_cache/tree/eviction.h"
#include "radix_cache/tree/node_ref.h"
#include "radix_cache/tree/radix_tree.h"
#include "radix_cache/tree/tree_node.h"
#include "radix_cache/unit_allocator.h"
#include "phyai_ext/radix_cache/tier.h"

#include <tvm/ffi/container/tensor.h>

namespace phyai_ext::radix_cache {

// Per-tier configuration accepted at construction.
struct tier_config {
  bool enabled = false;
  std::int32_t total_units = 0;
  bool is_async = false;  // hint for slow-tier (Disk/Remote); not enforced.
  // Upper bound on units that can be in the Pending state at the same time.
  // 0 = no limit. Used to prevent slow async I/O from starving the destination
  // tier with reservations whose data has not yet committed.
  std::int32_t max_pending_units = 0;
};

// Configuration for the prefix_cache constructor. Optional knobs default to
// sensible values; only `atom_bytes`, `atoms_per_unit` and one tier are
// strictly required.
struct prefix_cache_config {
  std::uint32_t atom_bytes = 0;
  std::uint32_t atoms_per_unit = 0;
  std::array<tier_config, max_tiers> tiers{};
  std::string eviction_policy = "lru";
  std::int64_t slru_threshold = 2;

  // Soft cap on the events buffer. When the buffer would exceed this size, the
  // oldest event is dropped and `dropped_events()` is incremented. 0 = no cap.
  std::size_t max_events_buffered = 16384;
};

// Observer hook invoked for every KV eviction. Used by the hybrid cache to
// detach an associated mamba slot, but consumers may register multiple
// observers for telemetry / consistency checks.
using evict_observer = std::function<void(tree_node*, tier)>;

// Strongly typed exception classes. tvm-ffi exposes them to Python as
// `phyai_ext.radix_cache.CacheCapacityError` etc. (see ffi_binding.cpp).
class cache_capacity_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class cache_usage_error : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};
class cache_invariant_error : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

class prefix_cache {
 public:
  using clock = std::chrono::steady_clock;

  prefix_cache(std::uint32_t atom_bytes, std::uint32_t atoms_per_unit, std::array<tier_config, max_tiers> tiers,
               std::string eviction_policy_name = "lru");
  explicit prefix_cache(prefix_cache_config cfg);

  prefix_cache(const prefix_cache&) = delete;
  prefix_cache& operator=(const prefix_cache&) = delete;

  // ── Query ──
  // Lightweight match: returns only the deepest matched node and the matched
  // atom count for each tier. The actual unit-id list is intentionally NOT
  // collected here; callers that need the ids should call `collect_units` /
  // `collect_units_dlpack` to materialise them lazily into a contiguous
  // buffer or DLPack tensor.
  struct match_result {
    std::array<tree_node*, max_tiers> last_node{};
    std::array<std::uint32_t, max_tiers> matched_atoms{};
  };
  match_result match(atom_span query);

  // Walk root -> `end` and collect every unit id stored on this tier into
  // `out` (caller-provided buffer). Returns the number of ids written; if
  // capacity is too small returns the required size without writing.
  std::size_t collect_units_into(tree_node* end, tier t, std::int32_t* out, std::size_t capacity) const;

  // Same as `collect_units_into` but returns the ids in a freshly-allocated
  // CPU int32 DLPack tensor. The tensor owns its buffer (allocated via
  // ::std::malloc) and is released through the standard DLPack deleter when
  // the last reference drops.
  tvm::ffi::Tensor collect_units_dlpack(tree_node* end, tier t) const;

  // ── Write ──
  struct insert_result {
    tree_node* last_node = nullptr;
    std::uint32_t inserted_atoms = 0;
    std::uint32_t freed_units_due_to_overlap = 0;
  };
  // Insert atoms into the tree backed by `units` for the given tier. Overlap
  // (units already in the tree) is freed automatically; remaining units are
  // attached to the suffix chain.
  insert_result insert(tier t, atom_span atoms, owned_units&& units);

  // ── Capacity ──
  // Evict from `t` until at least need_units are available. If promote_to is
  // set, evicted units are demoted to that tier (asynchronously, see the
  // OpHandle path); otherwise dropped.
  //
  // When the destination tier is also full, the call cascades: it first
  // evicts enough host (or whatever promote_to is) to make room, then resumes
  // the original eviction.
  void ensure_capacity(tier t, std::int32_t need_units, std::optional<tier> promote_to = std::nullopt);

  // Evict any node where `should_evict(node)` returns true. Returns the unit
  // count freed.
  std::uint32_t evict_by_predicate(tier t, const std::function<bool(const tree_node&)>& should_evict);

  // Native fast-path predicates, registered by short string keys. Avoids the
  // per-candidate Python callback overhead for common cases (SWA window
  // cutoff, hit-count threshold, …).
  std::uint32_t evict_by_named_predicate(tier t, const std::string& predicate, std::int64_t arg);

  // ── Locking ──
  node_ref lock(tier t, tree_node* node) { return node_ref(node, t); }
  composite_node_ref lock_multi(tree_node* node, std::initializer_list<tier> tiers) { return composite_node_ref(node, tiers); }

  // ── Allocation ──
  owned_units allocate(tier t, std::int32_t n);
  std::int32_t available(tier t) const;
  std::int32_t total(tier t) const;
  std::int32_t active(tier t) const;
  std::int32_t pending_units(tier t) const;
  std::int32_t max_pending_units(tier t) const noexcept { return tiers_[tier_index(t)].max_pending_units; }
  bool tier_enabled(tier t) const noexcept { return tiers_[tier_index(t)].enabled; }

  // ── Async ops ──
  op_handle start_demote(tree_node* node, tier src_tier, tier dst_tier);
  op_handle start_promote(tree_node* node, tier src_tier, tier dst_tier);
  void complete_op(op_handle handle, bool success);
  std::vector<op_handle> inflight_ops() const;

  // Block until `handle` completes or the timeout elapses. Returns true if
  // the op completed (regardless of success/failure), false on timeout.
  bool wait_op(op_handle handle, std::int64_t timeout_ms);

  // Mark every in-flight op as failed and reset the destination-tier
  // allocators. Intended for crash recovery after a process restart that
  // discovers leftover ops from the previous run. Returns the number of ops
  // marked failed.
  std::uint32_t fail_all_inflight();

  // ── Observers ──
  std::size_t add_evict_observer(evict_observer obs);
  bool remove_evict_observer(std::size_t observer_id);

  // ── Events stream ──
  std::vector<cache_event> take_events();
  std::size_t dropped_events() const;

  // ── Misc ──
  std::uint32_t atom_bytes() const noexcept { return atom_bytes_; }
  std::uint32_t atoms_per_unit() const noexcept { return atoms_per_unit_; }
  std::uint32_t page_bytes() const noexcept { return atom_bytes_ * atoms_per_unit_; }
  radix_tree& tree() noexcept { return tree_; }
  const radix_tree& tree() const noexcept { return tree_; }
  std::string policy_name() const { return policy_->name(); }
  std::int64_t slru_threshold() const noexcept { return cfg_.slru_threshold; }

  std::uint64_t current_step() const noexcept { return current_step_; }
  void advance_step(std::uint64_t n) noexcept { current_step_ += n; }
  void touch_step(tree_node* node);

  // Compute a 64-bit content hash for a node's full path (root -> node).
  // Useful for cross-process keys (storage backend, remote KV store).
  std::uint64_t node_path_hash(const tree_node* node) const;

 private:
  void collect_evict_candidates(tier t, std::vector<tree_node*>& out) const;
  std::uint32_t evict_node_resource(tree_node* n, tier t, std::optional<tier> promote_to);
  void emit(cache_event ev);
  op_handle next_handle();
  void check_pending_budget(tier dst_tier, std::size_t need_units) const;

  prefix_cache_config cfg_;
  std::uint32_t atom_bytes_;
  std::uint32_t atoms_per_unit_;
  std::array<tier_config, max_tiers> tiers_;

  std::vector<std::unique_ptr<unit_allocator>> allocators_;
  radix_tree tree_;
  std::unique_ptr<eviction_policy> policy_;
  std::vector<cache_event> events_;
  std::size_t dropped_events_ = 0;
  std::uint64_t current_step_ = 0;

  struct pending_op {
    tree_node* node = nullptr;
    tier src_tier = tier::device;
    tier dst_tier = tier::host;
    bool is_promote = false;
    std::uint32_t unit_count = 0;
  };
  std::unordered_map<op_handle, pending_op> inflight_;
  std::array<std::int32_t, max_tiers> pending_units_per_tier_{};
  op_handle next_handle_ = 1;

  struct observer_entry {
    std::size_t id;
    evict_observer fn;
  };
  std::vector<observer_entry> observers_;
  std::size_t next_observer_id_ = 1;

  mutable std::mutex events_mtx_;
  mutable std::mutex op_state_mtx_;
  std::condition_variable op_done_cv_;
};

}  // namespace phyai_ext::radix_cache
