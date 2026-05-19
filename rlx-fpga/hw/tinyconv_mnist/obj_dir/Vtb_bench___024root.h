// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_bench.h for the primary calling header

#ifndef VERILATED_VTB_BENCH___024ROOT_H_
#define VERILATED_VTB_BENCH___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_bench__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_bench___024root final {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        CData/*0:0*/ tb_bench__DOT__clk;
        CData/*0:0*/ tb_bench__DOT__rst;
        CData/*0:0*/ tb_bench__DOT__start;
        CData/*0:0*/ tb_bench__DOT__done;
        CData/*0:0*/ tb_bench__DOT__in_we;
        CData/*7:0*/ tb_bench__DOT__in_din;
        CData/*0:0*/ tb_bench__DOT__counting;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a0_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a1_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a2_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a3_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a4_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a5_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a6_dout;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__a7_addr;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a7_dout;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__a8_dout;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l0_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l0_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l0_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l1_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l1_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l1_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l2_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l2_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l2_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l3_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l3_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l3_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l4_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l4_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l4_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l5_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l5_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l5_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l6_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l6_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l6_y_we;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l7_start;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l7_done;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__l7_y_we;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__stage;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__cstate;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__cnext;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a0__DOT____Vlvbound_h8104ec03__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a1__DOT____Vlvbound_hd9a27771__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a2__DOT____Vlvbound_hd9a27771__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a3__DOT____Vlvbound_ha38edf7c__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a4__DOT____Vlvbound_h8ea625ea__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a5__DOT____Vlvbound_h8ea625ea__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a6__DOT____Vlvbound_h99f3a4a9__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a7__DOT____Vlvbound_hb1bb03c6__0;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_a8__DOT____Vlvbound_h8978cab3__0;
        CData/*6:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_byte;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout;
        CData/*4:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh;
        CData/*4:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow;
        CData/*4:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__state;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__next;
    };
    struct {
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_relu1__DOT__state;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_relu1__DOT__next;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__state;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__next;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_pool1__DOT__best;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_byte;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i;
        CData/*1:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__state;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__next;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_relu2__DOT__state;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_relu2__DOT__next;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__state;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__next;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i;
        CData/*0:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_pool2__DOT__best;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__w_byte;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__state;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__next;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_argmax__DOT__state;
        CData/*2:0*/ tb_bench__DOT__u_top__DOT__u_argmax__DOT__next;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_argmax__DOT__i;
        CData/*3:0*/ tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx;
        CData/*7:0*/ tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __VstlPhaseResult;
        CData/*0:0*/ __Vtrigprevexpr___TOP__tb_bench__DOT__clk__0;
        CData/*0:0*/ __Vtrigprevexpr___TOP__tb_bench__DOT__done__0;
        CData/*0:0*/ __VactPhaseResult;
        CData/*0:0*/ __VinactPhaseResult;
        CData/*0:0*/ __VnbaPhaseResult;
        SData/*9:0*/ tb_bench__DOT__in_addr;
        SData/*9:0*/ tb_bench__DOT__u_top__DOT__a0_addr;
        SData/*12:0*/ tb_bench__DOT__u_top__DOT__a1_addr;
        SData/*12:0*/ tb_bench__DOT__u_top__DOT__a2_addr;
        SData/*10:0*/ tb_bench__DOT__u_top__DOT__a3_addr;
        SData/*10:0*/ tb_bench__DOT__u_top__DOT__a4_addr;
        SData/*10:0*/ tb_bench__DOT__u_top__DOT__a5_addr;
        SData/*8:0*/ tb_bench__DOT__u_top__DOT__a6_addr;
        SData/*12:0*/ tb_bench__DOT__u_top__DOT__u_relu1__DOT__i;
        SData/*10:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr;
        SData/*10:0*/ tb_bench__DOT__u_top__DOT__u_relu2__DOT__i;
        SData/*11:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr;
        SData/*8:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__b_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out;
    };
    struct {
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__b_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__b_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__acc;
        IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out;
        IData/*31:0*/ __VactIterCount;
        IData/*31:0*/ __VinactIterCount;
        IData/*31:0*/ __Vi;
        QData/*63:0*/ tb_bench__DOT__cycles_counter;
        VlUnpacked<CData/*7:0*/, 784> tb_bench__DOT__image_mem;
        VlUnpacked<CData/*7:0*/, 784> tb_bench__DOT__u_top__DOT__u_a0__DOT__mem;
        VlUnpacked<CData/*7:0*/, 5408> tb_bench__DOT__u_top__DOT__u_a1__DOT__mem;
        VlUnpacked<CData/*7:0*/, 5408> tb_bench__DOT__u_top__DOT__u_a2__DOT__mem;
        VlUnpacked<CData/*7:0*/, 1352> tb_bench__DOT__u_top__DOT__u_a3__DOT__mem;
        VlUnpacked<CData/*7:0*/, 1936> tb_bench__DOT__u_top__DOT__u_a4__DOT__mem;
        VlUnpacked<CData/*7:0*/, 1936> tb_bench__DOT__u_top__DOT__u_a5__DOT__mem;
        VlUnpacked<CData/*7:0*/, 400> tb_bench__DOT__u_top__DOT__u_a6__DOT__mem;
        VlUnpacked<CData/*7:0*/, 10> tb_bench__DOT__u_top__DOT__u_a7__DOT__mem;
        VlUnpacked<CData/*7:0*/, 1> tb_bench__DOT__u_top__DOT__u_a8__DOT__mem;
        VlUnpacked<CData/*7:0*/, 72> tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_w_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 8> tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_b_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 8> tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_m0_rom__DOT__mem;
        VlUnpacked<CData/*7:0*/, 8> tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_sh_rom__DOT__mem;
        VlUnpacked<CData/*7:0*/, 1152> tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_w_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 16> tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_b_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 16> tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_m0_rom__DOT__mem;
        VlUnpacked<CData/*7:0*/, 16> tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_sh_rom__DOT__mem;
        VlUnpacked<CData/*7:0*/, 4000> tb_bench__DOT__u_top__DOT__u_fc__DOT__u_w_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 10> tb_bench__DOT__u_top__DOT__u_fc__DOT__u_b_rom__DOT__mem;
        VlUnpacked<IData/*31:0*/, 10> tb_bench__DOT__u_top__DOT__u_fc__DOT__u_m0_rom__DOT__mem;
        VlUnpacked<CData/*7:0*/, 10> tb_bench__DOT__u_top__DOT__u_fc__DOT__u_sh_rom__DOT__mem;
        VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
        VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
        VlUnpacked<QData/*63:0*/, 1> __VactTriggeredAcc;
        VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    };
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_ha7a70230__0;
    VlTriggerScheduler __VtrigSched_h94de5d66__0;

    // INTERNAL VARIABLES
    Vtb_bench__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_bench___024root(Vtb_bench__Syms* symsp, const char* namep);
    ~Vtb_bench___024root();
    VL_UNCOPYABLE(Vtb_bench___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
