#pragma once

#include <initializer_list>
#include <vector>

#include "radix_cache/tree/tree_node.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

// RAII pin against eviction. On construction, walks node->root and locks the
// node_resource of `t` on every ancestor that has one. On destruction,
// reverse-walks unlocking. Cost is O(depth) per construction/destruction.
class node_ref {
 public:
  node_ref() = default;
  node_ref(tree_node* node, tier t) noexcept : node_(node), tier_(t) {
    for (tree_node* p = node_; p != nullptr; p = p->parent()) {
      if (auto* r = p->resource(tier_)) r->lock();
    }
  }
  ~node_ref() { release(); }

  node_ref(const node_ref&) = delete;
  node_ref& operator=(const node_ref&) = delete;
  node_ref(node_ref&& o) noexcept : node_(o.node_), tier_(o.tier_) { o.node_ = nullptr; }
  node_ref& operator=(node_ref&& o) noexcept {
    if (this != &o) {
      release();
      node_ = o.node_;
      tier_ = o.tier_;
      o.node_ = nullptr;
    }
    return *this;
  }

  tree_node* node() const noexcept { return node_; }
  tier get_tier() const noexcept { return tier_; }
  bool valid() const noexcept { return node_ != nullptr; }

 private:
  void release() noexcept {
    if (node_ == nullptr) return;
    for (tree_node* p = node_; p != nullptr; p = p->parent()) {
      if (auto* r = p->resource(tier_)) {
        if (r->ref_count() > 0) r->unlock();
      }
    }
    node_ = nullptr;
  }

  tree_node* node_ = nullptr;
  tier tier_ = tier::device;
};

// Pin a node across multiple tiers simultaneously. Each requested tier gets
// its own parent-chain lock; destruction releases all of them. Useful when
// the caller wants the host copy not to be evicted while the device copy is
// in use, so a future demote can be a no-copy attach.
class composite_node_ref {
 public:
  composite_node_ref() = default;

  composite_node_ref(tree_node* node, std::initializer_list<tier> tiers) {
    refs_.reserve(tiers.size());
    for (tier t : tiers) refs_.emplace_back(node, t);
  }

  composite_node_ref(tree_node* node, const std::vector<tier>& tiers) {
    refs_.reserve(tiers.size());
    for (tier t : tiers) refs_.emplace_back(node, t);
  }

  composite_node_ref(const composite_node_ref&) = delete;
  composite_node_ref& operator=(const composite_node_ref&) = delete;
  composite_node_ref(composite_node_ref&&) noexcept = default;
  composite_node_ref& operator=(composite_node_ref&&) noexcept = default;

  std::size_t size() const noexcept { return refs_.size(); }
  bool empty() const noexcept { return refs_.empty(); }
  bool valid() const noexcept {
    if (refs_.empty()) return false;
    for (const auto& r : refs_) {
      if (!r.valid()) return false;
    }
    return true;
  }
  tree_node* node() const noexcept { return refs_.empty() ? nullptr : refs_.front().node(); }

 private:
  std::vector<node_ref> refs_;
};

}  // namespace phyai_ext::radix_cache
