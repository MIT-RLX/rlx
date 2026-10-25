-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Models: any provider's model, opened by path or id and run by name. A graph
-- saved as an .rlxp package runs anywhere; a plugin's models (a downstream
-- collection's LLMs, or the reference plugin's demo:count) load through
-- RLX_MODEL_PLUGINS.
--   RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib lua examples/models.lua

local rlx = require("rlx")

local g = rlx.Graph.new({ name = "affine" })
local x = g:input("x", { 1, 2 })
g:set_outputs({ g:matmul(x, g:param("w", { 2, 2 })) })
local path = (os.getenv("TMPDIR") or ".") .. "/rlx-example-lua.rlxp"
rlx.save_model(g, path, { w = { 2, 0, 0, 3 } }, { model = { task = "graph", family = "affine", outputs = { "y" } } })

local m = rlx.Model.open(path, { options = { device = "cpu" } })
local info = m:info()
print(string.format("model: %s (%s) on %s", info.family, info.task, info.device))
local y = m:run({ x = { 1, 1 } })[1]
print(string.format("model run: %.3f %.3f", y[1], y[2]))
os.remove(path)

local lm = rlx.Model.open("demo:count")
local lm_info = lm:info()
print(string.format("plugin model: %s (%s)", lm_info.family, lm_info.task))
print("model text: " .. lm:generate_text("c", { options = { max_tokens = 4 } }))
print("model ids: " .. table.concat(lm:generate({ 2 }, { max_tokens = 3 }), " "))

-- By task: a chat, speech out and back in, embeddings.
print("chat: " .. lm:chat({ { role = "user", content = "c" } }, { options = { max_tokens = 4 } }))
local speech = rlx.Model.open("demo:voice"):synthesize("hello")
print(string.format("voice: %d samples at %d Hz", #speech.audio, speech.sample_rate))
print("heard: " .. rlx.Model.open("demo:speech"):transcribe(speech.audio, { sample_rate = speech.sample_rate }))
local values = {}
for _, row in ipairs(rlx.Model.open("demo:embed"):embed({ "hello", "sky" })) do
  for _, v in ipairs(row) do values[#values + 1] = string.format("%d", math.floor(v)) end
end
print("embed: " .. table.concat(values, " "))
