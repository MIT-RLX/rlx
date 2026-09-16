/* Exposes the RLX node C ABI to Swift. The header ships with the crate that
 * defines the ABI (crates/bindings/rlx-ffi/include), reached via
 * HEADER_SEARCH_PATHS so there is one copy, not two. */
#import "rlx_node.h"
