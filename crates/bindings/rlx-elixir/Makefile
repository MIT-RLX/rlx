# RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
# priv/rlx_nif.so: the NIF, against librlx_capi in RLX_CAPI_LIB_DIR.
ERTS_INCLUDE ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')
UNAME := $(shell uname)
# A packaged release bundles librlx_capi under native/<os>-<arch>/; a checkout
# uses the workspace build.
RLX_OS := $(if $(filter Darwin,$(UNAME)),macos,linux)
RLX_ARCH := $(if $(filter arm64 aarch64,$(shell uname -m)),arm64,x64)
RLX_CAPI_LIB_DIR ?= $(if $(wildcard native/$(RLX_OS)-$(RLX_ARCH)),$(abspath native/$(RLX_OS)-$(RLX_ARCH)),../../../target/debug)
LDFLAGS_NIF = $(if $(filter Darwin,$(UNAME)),-dynamiclib -undefined dynamic_lookup,-shared)

priv/rlx_nif.so: c_src/rlx_nif.c c_src/rlx_nif_support.c c_src/rlx_nif.h
	mkdir -p priv
	$(CC) -O2 -fPIC -I$(ERTS_INCLUDE) -Ic_src $(LDFLAGS_NIF) c_src/rlx_nif.c c_src/rlx_nif_support.c \
		-L$(RLX_CAPI_LIB_DIR) -lrlx_capi -Wl,-rpath,$(RLX_CAPI_LIB_DIR) -o $@
