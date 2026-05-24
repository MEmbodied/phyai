#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "radix_cache/async/op_handle.h"
#include "radix_cache/owned_units.h"

namespace phyai_ext::radix_cache {

// Abstract storage backend used by slow tiers (disk, remote KV store).
//
// The backend is given a content-addressed key (typically the 64-bit hash of
// the path-prefix bytes) and a set of unit ids; it is responsible for the
// actual physical I/O. The returned op_handle is the same handle used by the
// prefix_cache state machine: when the I/O completes the backend calls
// `on_complete(handle, success)` (provided at registration) so the cache can
// transition the resource Pending -> Ready.
//
// Implementations of this interface are NOT required to be thread-safe by
// themselves; the prefix_cache serialises calls to the backend through its
// own scheduler thread. An implementation that uses a worker pool internally
// must serialise its own `on_complete` callbacks.
class storage_backend {
 public:
  using completion_callback = std::function<void(op_handle, bool)>;

  virtual ~storage_backend() = default;

  // Identifier for telemetry / error messages.
  virtual std::string name() const = 0;

  // Bytes per cache unit on the underlying medium. Used by the cache layer
  // when computing budgets; backends that don't know (e.g. an opaque KV
  // store) may return 0.
  virtual std::size_t unit_bytes() const = 0;

  // Begin writing the given units' contents to storage under `key`. The
  // backend must call `cb(handle, success)` exactly once when done.
  virtual void start_write(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) = 0;

  // Begin loading units associated with `key` into the destination ids.
  virtual void start_read(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) = 0;

  // Synchronous teardown: wait for all in-flight I/O to drain. Called when
  // the cache is being shut down or before crash-recovery resets state.
  virtual void drain() = 0;
};

// In-memory backend useful for tests and as a reference implementation. It
// keeps a hash -> bytes map and completes ops synchronously.
class in_memory_storage_backend final : public storage_backend {
 public:
  explicit in_memory_storage_backend(std::size_t unit_bytes) : unit_bytes_(unit_bytes) {}

  std::string name() const override { return "in_memory"; }
  std::size_t unit_bytes() const override { return unit_bytes_; }

  void start_write(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) override;
  void start_read(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) override;
  void drain() override {}

  bool contains(std::uint64_t key) const;
  std::size_t entries() const;

 private:
  std::size_t unit_bytes_;
  // Each entry stores an opaque blob (size = ids.size() * unit_bytes_) so
  // round-trip read can verify content. Real backends never see Python KV
  // tensors here — they only need ids; this implementation simply records the
  // ids it was given so tests can verify the backend was called correctly.
  struct entry {
    std::vector<unit_id> ids;
  };
  mutable std::mutex mtx_;
  std::unordered_map<std::uint64_t, entry> store_;
};

// File-backed backend. Each entry is stored as one record in a flat file; the
// caller supplies a path prefix. Designed for unit tests and small offline
// runs — production deployments would use io_uring or a managed KV store.
class file_storage_backend final : public storage_backend {
 public:
  file_storage_backend(std::string path, std::size_t unit_bytes);
  ~file_storage_backend() override;

  std::string name() const override { return "file"; }
  std::size_t unit_bytes() const override { return unit_bytes_; }

  void start_write(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) override;
  void start_read(op_handle handle, std::uint64_t key, std::span<const unit_id> ids, completion_callback cb) override;
  void drain() override;

 private:
  std::string path_;
  std::size_t unit_bytes_;
  int fd_ = -1;
  mutable std::mutex mtx_;
  std::unordered_map<std::uint64_t, std::int64_t> offsets_;  // key -> file offset
  std::int64_t end_offset_ = 0;
};

}  // namespace phyai_ext::radix_cache
