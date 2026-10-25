# frozen_string_literal: true

# RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.

Gem::Specification.new do |s|
  s.name = "rlx"
  s.version = "0.2.19"
  s.summary = "Ruby bindings for rlx over its C ABI (stdlib Fiddle)"
  s.description = "Graphs, sessions, device selection, autodiff, training, weights, text and " \
                  "distributed nodes from Ruby. Bundles librlx_capi for common platforms; RLX_CAPI_LIB overrides it."
  s.authors = ["Eugene Hauptmann", "Nataliya Kosmyna"]
  s.license = "MIT OR Apache-2.0"
  s.files = Dir["lib/**/*.rb"] + Dir["native/*/*"] + ["README.md"]
  # scripts/bindings_release.py builds one gem per platform, each carrying that
  # platform's librlx_capi under native/; the plain gem carries none.
  s.platform = ENV["RLX_GEM_PLATFORM"] if ENV["RLX_GEM_PLATFORM"]
  s.required_ruby_version = ">= 3.2"
  s.add_dependency "fiddle", ">= 1.1"
end
