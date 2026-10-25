-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

--- rlx for Lua 5.4+, over the `rlx_core` C module (src/rlx_core.c, generated
--- from rlx-bind's tables, plus src/rlx_lua_support.c). Every C function is a
--- Lua function or method: handle types are metatable classes freed by `__gc`
--- (or `<close>`), optional parameters a trailing options table
--- (`g:softmax(x, {axis = 1})`). Options and reports are Lua tables (JSON).

local core = require("rlx_core")

local M = {}
local S = { NONE = 0xFFFFFFFF }
M.NONE = S.NONE

local KINDS = { [-1] = "type", [-2] = "range", [-3] = "runtime", [-4] = "panic", [-5] = "not_found" }

--- A failed call raises `{kind = "range", message = "…"}` (tostring gives the message).
local Error = {}
Error.__index = Error
Error.__tostring = function(e) return "rlx " .. e.kind .. ": " .. e.message end
M.Error = Error

function S.check(rc)
  if rc ~= 0 then
    error(setmetatable({ kind = KINDS[rc] or "other", message = core.support_last_error() }, Error), 2)
  end
end

--- A class whose instances own a C handle, freed by `__gc`, `__close` or `:close()`.
function S.class(name, free)
  local cls = { __name = "rlx." .. name }
  cls.__index = cls
  local release = core[free]
  local function close(self)
    if self.handle then release(self.handle); self.handle = nil end
  end
  cls.close = close
  cls.__gc = close
  cls.__close = close
  return cls
end

function S.wrap(cls, handle)
  if handle == nil then error(setmetatable({ kind = "runtime", message = "rlx returned NULL" }, Error)) end
  return setmetatable({ handle = handle }, cls)
end

function S.rows(rows, w, splat)
  local out = {}
  for i, r in ipairs(rows) do
    if #r == w then for _, v in ipairs(r) do out[#out + 1] = v end
    elseif splat and #r == 1 then for _ = 1, w do out[#out + 1] = r[1] end
    else error(setmetatable({ kind = "type", message = ("row %d needs %d values, got %d"):format(i, w, #r) }, Error)) end
  end
  return out
end

local function sorted_keys(t)
  local keys = {}
  for k in pairs(t) do keys[#keys + 1] = k end
  table.sort(keys)
  return keys
end

function S.inputs(t)
  local names, data, lens = sorted_keys(t), {}, {}
  for i, k in ipairs(names) do data[i] = t[k]; lens[i] = #t[k] end
  return { names = names, data = data, lens = lens, n = #names }
end

function S.typed_inputs(t)
  local names, data, lens, dtypes = sorted_keys(t), {}, {}, {}
  for i, k in ipairs(names) do data[i] = t[k].data; lens[i] = #t[k].data; dtypes[i] = t[k].dtype end
  return { names = names, data = data, lens = lens, dtypes = dtypes, n = #names }
end

S.take_outputs = core.support_take_outputs
S.take_typed_outputs = core.support_take_typed_outputs
S.take_tensors = core.support_take_tensors

-- ── JSON (options in, reports out) ──────────────────────────

local json = {}
S.json = json
M.json = json

--- An array: a table whose keys are 1..n (an empty table encodes as `{}`).
local function is_array(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  return n > 0 and #t == n
end

function json.encode(v)
  local t = type(v)
  if v == nil then return "null"
  elseif t == "boolean" then return tostring(v)
  elseif t == "number" then
    if v ~= v or v == math.huge or v == -math.huge then return "null" end
    if math.type(v) == "integer" then return tostring(v) end
    return string.format("%.17g", v)
  elseif t == "string" then
    return '"' .. v:gsub('[%c"\\]', function(c)
      if c == '"' then return '\\"' elseif c == "\\" then return "\\\\" elseif c == "\n" then return "\\n" end
      return string.format("\\u%04x", c:byte())
    end) .. '"'
  elseif t == "table" then
    local parts = {}
    if is_array(v) then
      for _, x in ipairs(v) do parts[#parts + 1] = json.encode(x) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    for _, k in ipairs(sorted_keys(v)) do parts[#parts + 1] = json.encode(tostring(k)) .. ":" .. json.encode(v[k]) end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  error("not JSON-encodable: " .. t)
end

function json.decode(s)
  local i = 1
  local function ws() i = s:find("[^ \t\r\n]", i) or #s + 1 end
  local value
  local function str()
    local out, j = {}, i + 1
    while true do
      local c = s:sub(j, j)
      if c == '"' then i = j + 1; return table.concat(out) end
      if c == "\\" then
        local e = s:sub(j + 1, j + 1)
        if e == "u" then out[#out + 1] = utf8.char(tonumber(s:sub(j + 2, j + 5), 16)); j = j + 6
        else out[#out + 1] = ({ n = "\n", t = "\t", r = "\r", b = "\b", f = "\f" })[e] or e; j = j + 2 end
      else out[#out + 1] = c; j = j + 1 end
    end
  end
  function value()
    ws()
    local c = s:sub(i, i)
    if c == "{" then
      local t = {}; i = i + 1; ws()
      if s:sub(i, i) == "}" then i = i + 1; return t end
      while true do
        ws(); local k = str(); ws(); i = i + 1
        t[k] = value(); ws()
        local d = s:sub(i, i); i = i + 1
        if d == "}" then return t end
      end
    elseif c == "[" then
      local t = {}; i = i + 1; ws()
      if s:sub(i, i) == "]" then i = i + 1; return t end
      while true do
        t[#t + 1] = value(); ws()
        local d = s:sub(i, i); i = i + 1
        if d == "]" then return t end
      end
    elseif c == '"' then return str()
    elseif s:sub(i, i + 3) == "true" then i = i + 4; return true
    elseif s:sub(i, i + 4) == "false" then i = i + 5; return false
    elseif s:sub(i, i + 3) == "null" then i = i + 4; return nil
    end
    local num = s:match("^-?[%d.eE+-]+", i)
    i = i + #num
    return math.tointeger(tonumber(num)) or tonumber(num)
  end
  return value()
end

require("rlx.generated")(M, core, S)

-- ── the manual rows ─────────────────────────────────────────

--- Update each `params[name]` (a table of numbers) in place from `grads[name]`.
function M.Optimizer:step(params, grads)
  local names = sorted_keys(params)
  local ps, gs = {}, {}
  for k, n in ipairs(names) do
    ps[k] = params[n]
    gs[k] = assert(grads[n], "Optimizer:step: no gradient for '" .. n .. "'")
  end
  S.check(core.support_optimizer_step(self.handle, names, ps, gs))
end

--- Train `steps` steps, pulling inputs from `batch(step)` (nil: data exhausted).
--- `opts`: `{lr = function(step) … end, on_step = function(step, loss) return keep_going end, every = 1}`.
function M.Trainer:run(steps, batch, opts)
  opts = opts or {}
  local every = math.max(opts.every or 1, 1)
  local start = self:steps()
  local losses, stopped, t0 = {}, false, os.clock()
  for i = 0, steps - 1 do
    local s = start + i
    local b = batch(s)
    if b == nil then break end
    if opts.lr then self:set_lr(opts.lr(s)) end
    local loss = self:step(b)
    losses[#losses + 1] = loss
    if opts.on_step and (i + 1) % every == 0 and opts.on_step(s, loss) == false then stopped = true; break end
  end
  local sum = 0
  for _, l in ipairs(losses) do sum = sum + l end
  return {
    steps = #losses, stopped = stopped, first_loss = losses[1], last_loss = losses[#losses],
    mean_loss = #losses > 0 and sum / #losses or nil,
    steps_per_second = #losses / math.max(os.clock() - t0, 1e-9), grad_norm = self:grad_norm(),
  }
end

return M
