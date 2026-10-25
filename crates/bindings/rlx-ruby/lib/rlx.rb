# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Ruby bindings for rlx over its C ABI (crates/bindings/rlx-capi), on the
# standard library's Fiddle — no gems. Every C function is a Ruby method
# (lib/rlx/generated.rb, from rlx-bind's tables): handle types are classes
# freed by finalizers, optional parameters are keywords.
#
# The library is ENV["RLX_CAPI_LIB"] when set, else the one bundled with the
# gem (native/<os>-<arch>/), else librlx_capi from the workspace's target/debug
# (a checkout). Calls release the GVL, so mesh ranks can be threads.

require "fiddle"
require "json"

module RLX
  LIB_PATH = ENV.fetch("RLX_CAPI_LIB") do
    host = RbConfig::CONFIG["host_os"]
    os, file = if host =~ /darwin/ then ["macos", "librlx_capi.dylib"]
               elsif host =~ /mswin|mingw/ then ["windows", "rlx_capi.dll"]
               else ["linux", "librlx_capi.so"]
               end
    arch = RbConfig::CONFIG["host_cpu"] =~ /arm64|aarch64/ ? "arm64" : "x64"
    bundled = File.expand_path("../native/#{os}-#{arch}/#{file}", __dir__)
    File.exist?(bundled) ? bundled : File.expand_path("../../../../target/debug/#{file}", __dir__)
  end
  LIB = Fiddle.dlopen(LIB_PATH)

  # "No node", for an optional node argument.
  NONE = 0xFFFF_FFFF

  # A failed call: +kind+ is :type, :range, :runtime, :panic or :not_found.
  class Error < StandardError
    attr_reader :kind

    def initialize(kind, message)
      @kind = kind
      super(message)
    end
  end

  KINDS = { -1 => :type, -2 => :range, -3 => :runtime, -4 => :panic, -5 => :not_found }.freeze

  SUPPORT = {
    rlx_last_error: [[], :VOIDP],
    rlx_string_free: [[:VOIDP], :VOID],
    rlx_f32s_release: [[:VOIDP], :VOID],
    rlx_u32s_release: [[:VOIDP], :VOID],
    rlx_bytes_release: [[:VOIDP], :VOID],
    rlx_outputs_len: [[:VOIDP], :SIZE_T],
    rlx_outputs_get: [%i[VOIDP SIZE_T VOIDP VOIDP], :INT32_T],
    rlx_outputs_free: [[:VOIDP], :VOID],
    rlx_typed_outputs_len: [[:VOIDP], :SIZE_T],
    rlx_typed_outputs_get: [%i[VOIDP SIZE_T VOIDP VOIDP VOIDP], :INT32_T],
    rlx_typed_outputs_free: [[:VOIDP], :VOID],
    rlx_tensor_map_len: [[:VOIDP], :SIZE_T],
    rlx_tensor_map_get: [%i[VOIDP SIZE_T VOIDP VOIDP VOIDP], :INT32_T],
    rlx_tensor_map_free: [[:VOIDP], :VOID],
    rlx_optimizer_step: [%i[VOIDP VOIDP VOIDP VOIDP VOIDP VOIDP VOIDP SIZE_T], :INT32_T],
  }.freeze

  @functions = {}
  @lock = Mutex.new

  def self.function(name)
    @functions[name] || @lock.synchronize do
      @functions[name] ||= begin
        args, ret = SIGNATURES[name] || SUPPORT.fetch(name)
        Fiddle::Function.new(LIB[name.to_s], args.map { |t| Fiddle.const_get("TYPE_#{t}") },
                             Fiddle.const_get("TYPE_#{ret}"))
      end
    end
  end

  def self.call(name, *args)
    function(name).call(*args)
  end

  def self.last_error
    p = call(:rlx_last_error)
    p.null? ? "" : p.to_s
  end

  def self.check_rc(rc)
    return if rc.zero?

    raise Error.new(KINDS.fetch(rc, :unknown), last_error)
  end

  # ── arguments ──────────────────────────────────────────────

  def self.cstr(s)
    Fiddle::Pointer["#{s}\0"]
  end

  def self.buf(fmt, values)
    return Fiddle::Pointer[values] if fmt == "C" && values.is_a?(String)

    Fiddle::Pointer[values.to_a.pack("#{fmt}*")]
  end

  def self.rows(rows, width, splat)
    rows.flat_map do |r|
      r = Array(r)
      next r if r.size == width
      next Array.new(width, r[0]) if splat && r.size == 1

      raise Error.new(:type, "row needs #{width} values, got #{r.size}")
    end
  end

  # A list of C strings: +ptr+ to the pointer array, with the strings it points into.
  CStrings = Struct.new(:ptr, :size, :keep)

  def self.cstrs(list)
    return CStrings.new(nil, 0, nil) if list.nil?

    strs = list.map { |s| cstr(s) }
    CStrings.new(Fiddle::Pointer[strs.map(&:to_i).pack("J*")], strs.size, strs)
  end

  Inputs = Struct.new(:names, :data, :lens, :dtypes, :size, :keep)

  # Named f32 tensors ({"x" => [..]}) as the parallel C arrays a run takes.
  def self.inputs(hash)
    pairs = hash.to_a.sort_by { |k, _| k.to_s }
    names = pairs.map { |k, _| cstr(k) }
    data = pairs.map { |_, v| buf("f", v) }
    Inputs.new(Fiddle::Pointer[names.map(&:to_i).pack("J*")], Fiddle::Pointer[data.map(&:to_i).pack("J*")],
               Fiddle::Pointer[pairs.map { |_, v| v.size }.pack("J*")], nil, pairs.size, [names, data])
  end

  # A typed tensor: raw bytes and a dtype name ("f32", "i64", …).
  TypedTensor = Struct.new(:bytes, :dtype)

  def self.typed_inputs(hash)
    pairs = hash.to_a.sort_by { |k, _| k.to_s }
    names = pairs.map { |k, _| cstr(k) }
    data = pairs.map { |_, t| Fiddle::Pointer[t.bytes] }
    dtypes = pairs.map { |_, t| cstr(t.dtype) }
    Inputs.new(Fiddle::Pointer[names.map(&:to_i).pack("J*")], Fiddle::Pointer[data.map(&:to_i).pack("J*")],
               Fiddle::Pointer[pairs.map { |_, t| t.bytes.bytesize }.pack("J*")],
               Fiddle::Pointer[dtypes.map(&:to_i).pack("J*")], pairs.size, [names, data, dtypes])
  end

  # ── results ────────────────────────────────────────────────

  def self.out(bytes)
    Fiddle::Pointer.malloc(bytes, Fiddle::RUBY_FREE).tap { |p| p[0, bytes] = "\0" * bytes }
  end

  def self.ptr_at(out)
    Fiddle::Pointer.new(out[0, 8].unpack1("J"))
  end

  def self.take_string(out)
    p = ptr_at(out)
    return "" if p.null?

    s = p.to_s.force_encoding(Encoding::UTF_8)
    call(:rlx_string_free, p)
    s
  end

  def self.take_string_or_nil(out)
    ptr_at(out).null? ? nil : take_string(out)
  end

  def self.take_array(out, fmt, size, release)
    p = ptr_at(out)
    len = out[8, 8].unpack1("J")
    values = p.null? || len.zero? ? [] : p[0, len * size].unpack("#{fmt}*")
    call(release, out)
    fmt == "C" ? values.pack("C*") : values
  end

  def self.take_outputs(out)
    o = ptr_at(out)
    n = call(:rlx_outputs_len, o)
    result = Array.new(n) do |i|
      d = out(8)
      l = out(8)
      check_rc(call(:rlx_outputs_get, o, i, d, l))
      len = l[0, 8].unpack1("J")
      len.zero? ? [] : ptr_at(d)[0, len * 4].unpack("f*")
    end
    call(:rlx_outputs_free, o)
    result
  end

  def self.take_typed_outputs(out)
    o = ptr_at(out)
    n = call(:rlx_typed_outputs_len, o)
    result = Array.new(n) do |i|
      d = out(8)
      l = out(8)
      t = out(8)
      check_rc(call(:rlx_typed_outputs_get, o, i, d, l, t))
      len = l[0, 8].unpack1("J")
      TypedTensor.new(len.zero? ? "".b : ptr_at(d)[0, len], ptr_at(t).to_s)
    end
    call(:rlx_typed_outputs_free, o)
    result
  end

  def self.take_tensors(out)
    m = ptr_at(out)
    n = call(:rlx_tensor_map_len, m)
    result = (0...n).to_h do |i|
      nm = out(8)
      d = out(8)
      l = out(8)
      check_rc(call(:rlx_tensor_map_get, m, i, nm, d, l))
      len = l[0, 8].unpack1("J")
      [ptr_at(nm).to_s, len.zero? ? [] : ptr_at(d)[0, len * 4].unpack("f*")]
    end
    call(:rlx_tensor_map_free, m)
    result
  end

  # ── handles ────────────────────────────────────────────────

  # The base of every handle class: owns the C pointer, frees it when collected.
  class Handle
    attr_reader :handle

    def self.wrap(ptr)
      allocate.tap { |h| h.send(:adopt, ptr) }
    end

    def self.releaser(free, ptr)
      proc { RLX.call(free, ptr) }
    end

    def inspect
      "#<RLX::#{self.class.name.split('::').last}>"
    end

    private

    def adopt(ptr)
      raise Error.new(:runtime, "rlx returned a NULL #{self.class}") if ptr.null?

      @handle = ptr
      ObjectSpace.define_finalizer(self, Handle.releaser(self.class::FREE, ptr))
    end
  end
end

require_relative "rlx/generated"

module RLX
  class Optimizer
    # Update each params[name] (an Array of floats) in place from grads[name].
    def step(params, grads)
      names = params.keys.sort_by(&:to_s)
      c_names = RLX.cstrs(names)
      bufs = names.map { |n| RLX.buf("f", params[n]) }
      g = names.map { |n| RLX.buf("f", grads.fetch(n)) }
      RLX.check_rc(RLX.call(:rlx_optimizer_step, @handle, c_names.ptr,
                         Fiddle::Pointer[bufs.map(&:to_i).pack("J*")], Fiddle::Pointer[g.map(&:to_i).pack("J*")],
                         Fiddle::Pointer[names.map { |n| params[n].size }.pack("J*")], nil, nil, names.size))
      names.each_with_index { |n, i| params[n].replace(bufs[i][0, params[n].size * 4].unpack("f*")) }
      params
    end
  end

  class Trainer
    # Train +steps+ steps: the block (or +batch+) returns step +s+'s inputs, or
    # nil when the data is exhausted; +lr+ sets each step's rate; +on_step+
    # returns false to stop. Returns the run summary.
    def run(steps, batch: nil, lr: nil, on_step: nil, every: 1, &block)
      batch ||= block
      losses = []
      stopped = false
      t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      start = self.steps
      steps.times do |i|
        s = start + i
        b = batch.call(s)
        break if b.nil?

        set_lr(lr.call(s)) if lr
        loss = step(b)
        losses << loss
        next unless on_step && ((i + 1) % every).zero? && on_step.call(s, loss) == false

        stopped = true
        break
      end
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
      { "steps" => losses.size, "stopped" => stopped, "first_loss" => losses.first, "last_loss" => losses.last,
        "mean_loss" => losses.empty? ? nil : losses.sum / losses.size,
        "steps_per_second" => losses.size / [elapsed, 1e-9].max, "grad_norm" => grad_norm }
    end
  end
end
