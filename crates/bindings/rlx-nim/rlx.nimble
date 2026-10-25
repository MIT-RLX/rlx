# RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
version       = "0.2.19"
author        = "Eugene Hauptmann, Nataliya Kosmyna"
description   = "Nim bindings for rlx over its C ABI: graphs, sessions, device selection, autodiff, training, weights, text and distributed nodes."
license       = "MIT OR Apache-2.0"
srcDir        = "src"

requires "nim >= 2.0.0"

task test, "Run the tests":
  exec "nim r --hints:off tests/test_rlx.nim"
