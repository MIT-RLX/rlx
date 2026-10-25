// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated rlx.hpp stands on: the error type, the RAII handle base,
// the tensor carriers, the C-array builders, taking ownership of what rlx
// returns, and the two rows whose shape is not a plain call (`Optimizer::step`
// updates the caller's arrays in place; `Trainer::run` takes callbacks).

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "rlx.h"
}

namespace rlx {

/// A node of a graph — the id every builder returns.
using Node = uint32_t;
/// "No node", for an optional node argument.
inline constexpr Node NONE = 0xFFFFFFFFu;

/// A failed rlx call: `kind()` is the `RLX_ERR_*` status.
class Error : public std::runtime_error {
public:
    enum class Kind { Type = -1, Range = -2, Runtime = -3, Panic = -4, NotFound = -5, Other = 0 };
    Error(Kind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    Kind kind() const { return kind_; }

private:
    Kind kind_;
};

/// A tensor of any dtype as raw bytes (`"f32"`, `"i64"`, …).
struct TypedTensor {
    std::vector<uint8_t> bytes;
    std::string dtype;
};

/// Named f32 tensors: a run's inputs, parameters.
using Inputs = std::map<std::string, std::vector<float>>;
/// Named typed tensors.
using TypedInputs = std::map<std::string, TypedTensor>;
/// Named f32 tensors rlx returned (weights, trainer params).
using Tensors = std::map<std::string, std::vector<float>>;

namespace detail {

inline void check(int32_t rc) {
    if (rc == RLX_OK) return;
    auto kind = rc >= -5 && rc <= -1 ? static_cast<Error::Kind>(rc) : Error::Kind::Other;
    throw Error(kind, rlx_last_error());
}

/// Owns one rlx handle: move-only, freed on destruction.
template <class T, void (*Free)(T*)>
class Handle {
public:
    explicit Handle(T* h) : h_(h) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            reset();
            h_ = std::exchange(o.h_, nullptr);
        }
        return *this;
    }
    ~Handle() { reset(); }

    /// The C handle, for calling rlx.h directly.
    T* raw() const { return h_; }
    /// Free the handle now.
    void reset() {
        if (h_) Free(h_);
        h_ = nullptr;
    }

protected:
    T* h_;
};

/// Rows flattened; with `splat` a one-value row stands for `w` copies.
inline std::vector<size_t> rows(const std::vector<std::vector<size_t>>& rows, size_t w, bool splat) {
    std::vector<size_t> flat;
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].size() == w) flat.insert(flat.end(), rows[i].begin(), rows[i].end());
        else if (splat && rows[i].size() == 1) flat.insert(flat.end(), w, rows[i][0]);
        else throw Error(Error::Kind::Type, "row " + std::to_string(i) + " needs " + std::to_string(w) + " values");
    }
    return flat;
}

inline std::vector<float> frows(const std::vector<std::vector<float>>& rows, size_t w) {
    std::vector<float> flat;
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].size() != w) throw Error(Error::Kind::Type, "row " + std::to_string(i) + " needs " + std::to_string(w) + " values");
        flat.insert(flat.end(), rows[i].begin(), rows[i].end());
    }
    return flat;
}

/// A list of strings as `const char* const*`.
class CStrings {
public:
    explicit CStrings(const std::vector<std::string>& v) { for (auto& s : v) ptrs_.push_back(s.c_str()); has_ = true; }
    explicit CStrings(const std::optional<std::vector<std::string>>& v) {
        if (v) { for (auto& s : *v) ptrs_.push_back(s.c_str()); has_ = true; }
    }
    const char* const* data() const { return has_ ? ptrs_.data() : nullptr; }
    size_t size() const { return ptrs_.size(); }

private:
    std::vector<const char*> ptrs_;
    bool has_ = false;
};

class CInputs {
public:
    explicit CInputs(const Inputs& m) {
        for (auto& [k, v] : m) { names_.push_back(k.c_str()); data_.push_back(v.data()); lens_.push_back(v.size()); }
    }
    const char* const* names() const { return names_.data(); }
    const float* const* data() const { return data_.data(); }
    const size_t* lens() const { return lens_.data(); }
    size_t size() const { return names_.size(); }

private:
    std::vector<const char*> names_;
    std::vector<const float*> data_;
    std::vector<size_t> lens_;
};

class CTypedInputs {
public:
    explicit CTypedInputs(const TypedInputs& m) {
        for (auto& [k, t] : m) {
            names_.push_back(k.c_str()); data_.push_back(t.bytes.data());
            lens_.push_back(t.bytes.size()); dtypes_.push_back(t.dtype.c_str());
        }
    }
    const char* const* names() const { return names_.data(); }
    const uint8_t* const* data() const { return data_.data(); }
    const size_t* lens() const { return lens_.data(); }
    const char* const* dtypes() const { return dtypes_.data(); }
    size_t size() const { return names_.size(); }

private:
    std::vector<const char*> names_;
    std::vector<const uint8_t*> data_;
    std::vector<size_t> lens_;
    std::vector<const char*> dtypes_;
};

inline std::string take_string(char* p) {
    if (!p) return {};
    std::string s(p);
    rlx_string_free(p);
    return s;
}

inline std::optional<std::string> take_optional_string(char* p) {
    if (!p) return std::nullopt;
    return take_string(p);
}

inline std::vector<float> take_f32s(RlxF32s v) {
    std::vector<float> out(v.ptr, v.ptr + v.len);
    rlx_f32s_free(v);
    return out;
}

inline std::vector<uint8_t> take_bytes(RlxBytes v) {
    std::vector<uint8_t> out(v.ptr, v.ptr + v.len);
    rlx_bytes_free(v);
    return out;
}

inline std::vector<uint32_t> take_u32s(RlxU32s v) {
    std::vector<uint32_t> out(v.ptr, v.ptr + v.len);
    rlx_u32s_free(v);
    return out;
}

inline std::vector<std::vector<float>> take_outputs(RlxOutputs* o) {
    std::vector<std::vector<float>> out;
    for (size_t i = 0, n = rlx_outputs_len(o); i < n; i++) {
        const float* p = nullptr;
        size_t len = 0;
        if (rlx_outputs_get(o, i, &p, &len) != RLX_OK) { rlx_outputs_free(o); check(RLX_ERR_RUNTIME); }
        out.emplace_back(p, p + len);
    }
    rlx_outputs_free(o);
    return out;
}

inline std::vector<TypedTensor> take_typed_outputs(RlxTypedOutputs* o) {
    std::vector<TypedTensor> out;
    for (size_t i = 0, n = rlx_typed_outputs_len(o); i < n; i++) {
        const uint8_t* p = nullptr;
        size_t len = 0;
        const char* dt = nullptr;
        if (rlx_typed_outputs_get(o, i, &p, &len, &dt) != RLX_OK) { rlx_typed_outputs_free(o); check(RLX_ERR_RUNTIME); }
        out.push_back({std::vector<uint8_t>(p, p + len), dt});
    }
    rlx_typed_outputs_free(o);
    return out;
}

inline Tensors take_tensors(RlxTensorMap* m) {
    Tensors out;
    for (size_t i = 0, n = rlx_tensor_map_len(m); i < n; i++) {
        const char* name = nullptr;
        const float* p = nullptr;
        size_t len = 0;
        if (rlx_tensor_map_get(m, i, &name, &p, &len) != RLX_OK) { rlx_tensor_map_free(m); check(RLX_ERR_RUNTIME); }
        out.emplace(name, std::vector<float>(p, p + len));
    }
    rlx_tensor_map_free(m);
    return out;
}

/// A JSON array of strings (rlx's own output: no escapes beyond `\"` / `\\`).
inline std::vector<std::string> parse_strings(const std::string& json) {
    std::vector<std::string> out;
    for (size_t i = 0; i < json.size(); i++) {
        if (json[i] != '"') continue;
        std::string s;
        for (i++; i < json.size() && json[i] != '"'; i++) {
            if (json[i] == '\\' && i + 1 < json.size()) i++;
            s.push_back(json[i]);
        }
        out.push_back(std::move(s));
    }
    return out;
}

/// A JSON array of integer arrays (`[[2, 3], [4]]`).
inline std::vector<std::vector<int64_t>> parse_int_matrix(const std::string& json) {
    std::vector<std::vector<int64_t>> out;
    int depth = 0;
    std::string num;
    for (char c : json) {
        if (c == '[') { if (++depth == 2) out.emplace_back(); }
        else if (c == ']' || c == ',') {
            if (!num.empty()) { out.back().push_back(std::stoll(num)); num.clear(); }
            if (c == ']') depth--;
        } else if ((c >= '0' && c <= '9') || c == '-') num.push_back(c);
    }
    return out;
}

// ── the manual rows ─────────────────────────────────────────

/// `Optimizer::step`: update each `params[name]` in place from `grads[name]`.
template <class Self>
struct OptimizerManual {
    void step(Tensors& params, const Tensors& grads) {
        std::vector<const char*> names;
        std::vector<float*> ps;
        std::vector<const float*> gs;
        std::vector<size_t> lens;
        for (auto& [k, v] : params) {
            auto g = grads.find(k);
            if (g == grads.end() || g->second.size() != v.size())
                throw Error(Error::Kind::Type, "Optimizer::step: no matching gradient for '" + k + "'");
            names.push_back(k.c_str()); ps.push_back(v.data()); gs.push_back(g->second.data()); lens.push_back(v.size());
        }
        check(rlx_optimizer_step(static_cast<Self*>(this)->raw(), names.data(), ps.data(), gs.data(),
                                 lens.data(), nullptr, nullptr, names.size()));
    }
};

/// What `Trainer::run` reports.
struct RunSummary {
    size_t steps = 0;
    bool stopped = false;
    float first_loss = NAN, last_loss = NAN;
    double mean_loss = NAN, steps_per_second = 0;
    std::optional<float> grad_norm;
};

/// Options for `Trainer::run`.
struct RunOpts {
    /// The learning rate per step (unset: unchanged).
    std::function<float(uint64_t)> lr;
    /// Called every `every` steps with the loss; `false` stops.
    std::function<bool(uint64_t, float)> on_step;
    size_t every = 1;
};

/// `Trainer::run`: `steps` steps pulling inputs from `batch` (`nullopt` when the data is exhausted).
template <class Self>
struct TrainerManual {
    RunSummary run(size_t steps, const std::function<std::optional<Inputs>(uint64_t)>& batch, const RunOpts& opts = {}) {
        auto& self = *static_cast<Self*>(this);
        RunSummary r;
        double sum = 0;
        auto t0 = std::chrono::steady_clock::now();
        uint64_t start = self.steps();
        size_t every = opts.every ? opts.every : 1;
        for (size_t i = 0; i < steps; i++) {
            uint64_t s = start + i;
            auto b = batch(s);
            if (!b) break;
            if (opts.lr) self.set_lr(opts.lr(s));
            float loss = self.step(*b);
            if (r.steps == 0) r.first_loss = loss;
            r.last_loss = loss;
            sum += loss;
            r.steps++;
            if (opts.on_step && (i + 1) % every == 0 && !opts.on_step(s, loss)) { r.stopped = true; break; }
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r.mean_loss = r.steps ? sum / r.steps : NAN;
        r.steps_per_second = r.steps / (secs > 1e-9 ? secs : 1e-9);
        r.grad_norm = self.grad_norm();
        return r;
    }
};

}  // namespace detail
}  // namespace rlx
