// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_bench.h for the primary calling header

#include "Vtb_bench__pch.h"

VlCoroutine Vtb_bench___024root___eval_initial__TOP__Vtiming__0(Vtb_bench___024root* vlSelf);
VlCoroutine Vtb_bench___024root___eval_initial__TOP__Vtiming__1(Vtb_bench___024root* vlSelf);

void Vtb_bench___024root___eval_initial(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_initial\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 72, 0, "weights/conv1_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 8, 0, "weights/conv1_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 8, 0, "weights/conv1_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 8, 0, "weights/conv1_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 1152, 0, "weights/conv2_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "weights/conv2_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "weights/conv2_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 16, 0, "weights/conv2_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4000, 0, "weights/fc_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "weights/fc_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "weights/fc_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 10, 0, "weights/fc_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
    Vtb_bench___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_bench___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

void Vtb_bench___024root____VbeforeTrig_ha7a70230__0(Vtb_bench___024root* vlSelf, const char* __VeventDescription);
void Vtb_bench___024root____VbeforeTrig_h94de5d66__0(Vtb_bench___024root* vlSelf, const char* __VeventDescription);

VlCoroutine Vtb_bench___024root___eval_initial__TOP__Vtiming__0(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_bench__DOT__unnamedblk1__DOT__i;
    tb_bench__DOT__unnamedblk1__DOT__i = 0;
    // Body
    VL_READMEM_N(true, 8, 784, 0, "tb_image.mem"s,  &(vlSelfRef.tb_bench__DOT__image_mem)
                 , 0, ~0ULL);
    vlSelfRef.tb_bench__DOT__rst = 1U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000004e20ULL, 
                                         nullptr, "tb_bench.sv", 
                                         34);
    vlSelfRef.tb_bench__DOT__rst = 0U;
    tb_bench__DOT__unnamedblk1__DOT__i = 0U;
    while (VL_GTS_III(32, 0x00000310U, tb_bench__DOT__unnamedblk1__DOT__i)) {
        Vtb_bench___024root____VbeforeTrig_ha7a70230__0(vlSelf, 
                                                        "@(posedge tb_bench.clk)");
        co_await vlSelfRef.__VtrigSched_ha7a70230__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge tb_bench.clk)", 
                                                             "tb_bench.sv", 
                                                             36);
        vlSelfRef.tb_bench__DOT__in_addr = (0x000003ffU 
                                            & tb_bench__DOT__unnamedblk1__DOT__i);
        vlSelfRef.tb_bench__DOT__in_we = 1U;
        vlSelfRef.tb_bench__DOT__in_din = ((0x030fU 
                                            >= (0x000003ffU 
                                                & tb_bench__DOT__unnamedblk1__DOT__i))
                                            ? vlSelfRef.tb_bench__DOT__image_mem
                                           [(0x000003ffU 
                                             & tb_bench__DOT__unnamedblk1__DOT__i)]
                                            : 0U);
        tb_bench__DOT__unnamedblk1__DOT__i = ((IData)(1U) 
                                              + tb_bench__DOT__unnamedblk1__DOT__i);
    }
    Vtb_bench___024root____VbeforeTrig_ha7a70230__0(vlSelf, 
                                                    "@(posedge tb_bench.clk)");
    co_await vlSelfRef.__VtrigSched_ha7a70230__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_bench.clk)", 
                                                         "tb_bench.sv", 
                                                         41);
    vlSelfRef.tb_bench__DOT__in_we = 0U;
    Vtb_bench___024root____VbeforeTrig_ha7a70230__0(vlSelf, 
                                                    "@(posedge tb_bench.clk)");
    co_await vlSelfRef.__VtrigSched_ha7a70230__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_bench.clk)", 
                                                         "tb_bench.sv", 
                                                         42);
    vlSelfRef.tb_bench__DOT__start = 1U;
    while ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__done)))) {
        Vtb_bench___024root____VbeforeTrig_h94de5d66__0(vlSelf, 
                                                        "@( tb_bench.done)");
        co_await vlSelfRef.__VtrigSched_h94de5d66__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( tb_bench.done)", 
                                                             "tb_bench.sv", 
                                                             43);
    }
    Vtb_bench___024root____VbeforeTrig_ha7a70230__0(vlSelf, 
                                                    "@(posedge tb_bench.clk)");
    co_await vlSelfRef.__VtrigSched_ha7a70230__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge tb_bench.clk)", 
                                                         "tb_bench.sv", 
                                                         44);
    vlSelfRef.tb_bench__DOT__start = 0U;
    VL_WRITEF_NX("RESULT pred=%0d cycles=%0d\n",2, '~',8,(IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a8_dout)
                 , '~',64,vlSelfRef.tb_bench__DOT__cycles_counter);
    VL_FINISH_MT("tb_bench.sv", 46, "");
    co_return;
}

VlCoroutine Vtb_bench___024root___eval_initial__TOP__Vtiming__1(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x0000000000001388ULL, 
                                             nullptr, 
                                             "tb_bench.sv", 
                                             6);
        vlSelfRef.tb_bench__DOT__clk = (1U & (~ (IData)(vlSelfRef.tb_bench__DOT__clk)));
    }
    co_return;
}

void Vtb_bench___024root___eval_triggers_vec__act(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_triggers_vec__act\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                    ((((IData)(vlSelfRef.tb_bench__DOT__done) 
                                                       != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__done__0)) 
                                                      << 2U) 
                                                     | ((vlSelfRef.__VdlySched.awaitingCurrentTime() 
                                                         << 1U) 
                                                        | ((IData)(vlSelfRef.tb_bench__DOT__clk) 
                                                           & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0)))))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0 
        = vlSelfRef.tb_bench__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__done__0 
        = vlSelfRef.tb_bench__DOT__done;
}

bool Vtb_bench___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___trigger_anySet__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtb_bench___024root___act_comb__TOP__0(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___act_comb__TOP__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr = 
        (0x000003ffU & ((IData)(vlSelfRef.tb_bench__DOT__start)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic) 
                            + (((IData)(0x0000001cU) 
                                * ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i) 
                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh))) 
                               + ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow) 
                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))))
                         : (IData)(vlSelfRef.tb_bench__DOT__in_addr)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = vlSelfRef.tb_bench__DOT__u_top__DOT__cstate;
    if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
        if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__start)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 0U;
            }
        } else {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext 
                = ((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage))
                    ? 3U : 1U);
        }
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((1U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((5U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((6U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        }
    } else if (vlSelfRef.tb_bench__DOT__start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 1U;
    }
}

void Vtb_bench___024root___eval_act(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_act\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((7ULL & vlSelfRef.__VactTriggered[0U])) {
        Vtb_bench___024root___act_comb__TOP__0(vlSelf);
    }
}

extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_had1126da_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h21db8459_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h0e45ff86_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h392ef149_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h8ddecfd7_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hce4ff95b_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hd55a8f63_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hd25ea8a5_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hf85a0a8a_0;

void Vtb_bench___024root___nba_sequent__TOP__0(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___nba_sequent__TOP__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out = 0;
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out = 0;
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out = 0;
    CData/*5:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_0;
    __VdfgRegularize_h6e95ff9d_0_0 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_1;
    __VdfgRegularize_h6e95ff9d_0_1 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_2;
    __VdfgRegularize_h6e95ff9d_0_2 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_4;
    __VdfgRegularize_h6e95ff9d_0_4 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_6;
    __VdfgRegularize_h6e95ff9d_0_6 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_8;
    __VdfgRegularize_h6e95ff9d_0_8 = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__stage;
    __Vdly__tb_bench__DOT__u_top__DOT__stage = 0;
    CData/*4:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh = 0;
    CData/*4:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow = 0;
    CData/*4:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc = 0;
    CData/*1:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i = 0;
    CData/*1:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i = 0;
    CData/*0:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic = 0;
    IData/*31:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc = 0;
    SData/*12:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc = 0;
    CData/*0:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i = 0;
    CData/*0:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i = 0;
    CData/*7:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc = 0;
    CData/*1:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i = 0;
    CData/*1:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i = 0;
    CData/*2:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic = 0;
    IData/*31:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc = 0;
    SData/*10:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc = 0;
    CData/*0:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i = 0;
    CData/*0:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i = 0;
    CData/*7:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i = 0;
    SData/*8:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i = 0;
    IData/*31:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc = 0;
    CData/*3:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i = 0;
    CData/*7:0*/ __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val;
    __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 = 0;
    SData/*9:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 = 0;
    SData/*12:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 = 0;
    SData/*12:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 = 0;
    SData/*10:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 = 0;
    SData/*10:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 = 0;
    SData/*10:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 = 0;
    SData/*8:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 = 0;
    CData/*3:0*/ __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0;
    __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 = 0;
    CData/*7:0*/ __VdlyVal__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0;
    __VdlyVal__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0 = 0;
    CData/*0:0*/ __VdlySet__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0 = 0;
    // Body
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 = 0U;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 = 0U;
    __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val;
    __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i;
    __VdlySet__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0 = 0U;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__best;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__best;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__stage = vlSelfRef.tb_bench__DOT__u_top__DOT__stage;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh;
    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow;
    if (((~ (IData)(vlSelfRef.tb_bench__DOT__start)) 
         & (IData)(vlSelfRef.tb_bench__DOT__in_we))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a0__DOT____Vlvbound_h8104ec03__0 
            = ((IData)(vlSelfRef.tb_bench__DOT__in_din) 
               & (- (IData)((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__start))))));
        if ((0x030fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a0__DOT____Vlvbound_h8104ec03__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0 = 1U;
        }
    }
    if (((IData)(vlSelfRef.tb_bench__DOT__counting) 
         & (~ (IData)(vlSelfRef.tb_bench__DOT__done)))) {
        vlSelfRef.tb_bench__DOT__cycles_counter = (1ULL 
                                                   + vlSelfRef.tb_bench__DOT__cycles_counter);
    }
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l7_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a8__DOT____Vlvbound_h8978cab3__0 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx;
        __VdlyVal__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a8__DOT____Vlvbound_h8978cab3__0;
        __VdlySet__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0 = 1U;
    }
    if (vlSelfRef.tb_bench__DOT__rst) {
        __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best = 0x80U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best = 0x80U;
        __Vdly__tb_bench__DOT__u_top__DOT__stage = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx = 0U;
        __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val = 0x80U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__cstate = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state = 0U;
    } else {
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i = 0U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i 
                = (0x00001fffU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i)));
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i = 0U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i 
                = (0x000007ffU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i)));
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i = 0U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__b_dout;
        }
        if ((6U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc 
                = (vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc 
                   + VL_MULS_III(32, (((- (IData)((1U 
                                                   & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a6_dout) 
                                                      >> 7U)))) 
                                       << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a6_dout)), 
                                 (((- (IData)((1U & 
                                               ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_byte) 
                                                >> 7U)))) 
                                   << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_byte))));
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i 
                = ((0x018fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i))
                    ? 0U : (0x000001ffU & ((IData)(1U) 
                                           + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i))));
        }
        if ((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i 
                = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i)));
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best = 0x80U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
            if (VL_GTS_III(8, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a5_dout), (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__best))) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best 
                    = vlSelfRef.tb_bench__DOT__u_top__DOT__a5_dout;
            }
            if (vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i 
                    = ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i))) 
                       && (1U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i))));
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i 
                    = (1U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i)));
            }
        }
        if ((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
            if ((0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc))) {
                if ((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh)));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc 
                    = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc)));
            }
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best = 0x80U;
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best = 0x80U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
            if (VL_GTS_III(8, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a2_dout), (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__best))) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best 
                    = vlSelfRef.tb_bench__DOT__u_top__DOT__a2_dout;
            }
            if (vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i 
                    = ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i))) 
                       && (1U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i))));
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i 
                    = (1U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i)));
            }
        }
        if ((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
            if ((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc))) {
                if ((0x0cU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh)));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc 
                    = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc)));
            }
            __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best = 0x80U;
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate)) 
             & (IData)(vlSelfRef.tb_bench__DOT__start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__stage = 0U;
        }
        if ((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
            __Vdly__tb_bench__DOT__u_top__DOT__stage 
                = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)));
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic = 0U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__b_dout;
        }
        if ((6U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc 
                = (vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc 
                   + VL_MULS_III(32, (((- (IData)((1U 
                                                   & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a0_dout) 
                                                      >> 7U)))) 
                                       << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a0_dout)), 
                                 (((- (IData)((1U & 
                                               ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_byte) 
                                                >> 7U)))) 
                                   << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_byte))));
            if (vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic 
                    = (1U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic)));
            } else {
                if ((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i 
                        = ((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i))
                            ? 0U : (3U & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i))));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic = 0U;
            }
        }
        if ((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
            if ((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc))) {
                if ((0x19U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh 
                        = (0x0000001fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh)));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow 
                        = (0x0000001fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc 
                    = (0x0000001fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc)));
            }
        }
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic = 0U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__b_dout;
        }
        if ((6U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc 
                = (vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc 
                   + VL_MULS_III(32, (((- (IData)((1U 
                                                   & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a3_dout) 
                                                      >> 7U)))) 
                                       << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a3_dout)), 
                                 (((- (IData)((1U & 
                                               ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_byte) 
                                                >> 7U)))) 
                                   << 8U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_byte))));
            if ((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic))) {
                if ((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i 
                        = ((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i))
                            ? 0U : (3U & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i))));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i 
                        = (3U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic 
                    = (7U & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic)));
            }
        }
        if ((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
            if ((0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc))) {
                if ((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow))) {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh)));
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow = 0U;
                } else {
                    __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow 
                        = (0x0000000fU & ((IData)(1U) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow)));
                }
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc = 0U;
            } else {
                __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc 
                    = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc)));
            }
        }
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next;
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start))) {
            __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i = 0U;
            vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx = 0U;
            __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val = 0x80U;
        }
        if ((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
            if (VL_GTS_III(8, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a7_dout), (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val))) {
                __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val 
                    = vlSelfRef.tb_bench__DOT__u_top__DOT__a7_dout;
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx 
                    = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i;
            }
            __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i 
                = (0x0000000fU & ((IData)(1U) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i)));
        }
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__cstate 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__cnext;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a8_dout = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a8__DOT__mem[0U];
    if ((9U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_sh_rom__DOT__mem
            [vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i];
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_m0_rom__DOT__mem
            [vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i];
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__b_dout 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_b_rom__DOT__mem
            [vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i];
    } else {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout = 0U;
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__b_dout = 0U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_sh_rom__DOT__mem
        [(7U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc))];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_sh_rom__DOT__mem
        [vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_m0_rom__DOT__mem
        [(7U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc))];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_m0_rom__DOT__mem
        [vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_relu1__DOT__i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_relu2__DOT__i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__acc;
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a6__DOT____Vlvbound_h99f3a4a9__0 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__best;
        if ((0x018fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a6__DOT____Vlvbound_h99f3a4a9__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0 = 1U;
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__best 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__best;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc;
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a3__DOT____Vlvbound_ha38edf7c__0 
            = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__best;
        if ((0x0547U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a3__DOT____Vlvbound_ha38edf7c__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0 = 1U;
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__best 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__best;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc;
    vlSelfRef.tb_bench__DOT__u_top__DOT__stage = __Vdly__tb_bench__DOT__u_top__DOT__stage;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow;
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a8__DOT__mem[0U] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a8__DOT__mem__v0;
    }
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a2__DOT____Vlvbound_hd9a27771__0 
            = ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a1_dout) 
               & (- (IData)(VL_LTES_III(8, 0U, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a1_dout)))));
        if ((0x151fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a2__DOT____Vlvbound_hd9a27771__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0 = 1U;
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a1_dout = 
        ((0x151fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a1__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr]
          : 0U);
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a5__DOT____Vlvbound_h8ea625ea__0 
            = ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a4_dout) 
               & (- (IData)(VL_LTES_III(8, 0U, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a4_dout)))));
        if ((0x078fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a5__DOT____Vlvbound_h8ea625ea__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0 = 1U;
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a4_dout = 
        ((0x078fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a4__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr]
          : 0U);
    if (vlSelfRef.tb_bench__DOT__start) {
        vlSelfRef.tb_bench__DOT__counting = 1U;
    }
    if (vlSelfRef.tb_bench__DOT__done) {
        vlSelfRef.tb_bench__DOT__counting = 0U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_byte 
        = ((0x0f9fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr))
            ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_w_rom__DOT__mem
           [vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr]
            : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__a6_dout = 
        ((0x018fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a6__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__a5_dout = 
        ((0x078fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a5__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__a2_dout = 
        ((0x151fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a2__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_byte 
        = ((0x47U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr))
            ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_w_rom__DOT__mem
           [vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr]
            : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__a0_dout = 
        ((0x030fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a0__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__b_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_b_rom__DOT__mem
        [(7U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc))];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_byte 
        = ((0x047fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr))
            ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_w_rom__DOT__mem
           [vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr]
            : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__a3_dout = 
        ((0x0547U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a3__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__b_dout 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_b_rom__DOT__mem
        [vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc];
    __VdfgRegularize_h6e95ff9d_0_8 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout));
    __VdfgRegularize_h6e95ff9d_0_4 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout));
    __VdfgRegularize_h6e95ff9d_0_6 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout));
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a1__DOT____Vlvbound_hd9a27771__0 
            = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out)
                ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out)
                                  ? 0x00000080U : (0x000000ffU 
                                                   & vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out)));
        if ((0x151fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a1__DOT____Vlvbound_hd9a27771__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0 = 1U;
        }
    }
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a1__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a1__DOT__mem__v0;
    }
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a4__DOT____Vlvbound_h8ea625ea__0 
            = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out)
                ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out)
                                  ? 0x00000080U : (0x000000ffU 
                                                   & vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out)));
        if ((0x078fU >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a4__DOT____Vlvbound_h8ea625ea__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0 = 1U;
        }
    }
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a4__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a4__DOT__mem__v0;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_argmax__DOT__i;
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a6__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a6__DOT__mem__v0;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i;
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a5__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a5__DOT__mem__v0;
    }
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a2__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a2__DOT__mem__v0;
    }
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a0__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a0__DOT__mem__v0;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc;
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a3__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a3__DOT__mem__v0;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc 
        = __Vdly__tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc;
    __VdfgRegularize_h6e95ff9d_0_2 = (__VdfgRegularize_h6e95ff9d_0_8 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_8)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    __VdfgRegularize_h6e95ff9d_0_0 = (__VdfgRegularize_h6e95ff9d_0_4 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_4)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    __VdfgRegularize_h6e95ff9d_0_1 = (__VdfgRegularize_h6e95ff9d_0_6 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_6)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we = 0U;
    if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state) 
                  >> 2U)))) {
        if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr = 
        (0x00001fffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we)
                         ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i)
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc) 
                            + (0x00001ff8U & ((((IData)(0x0000001aU) 
                                                * (
                                                   VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh), 1U) 
                                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i))) 
                                               + (VL_SHIFTL_III(10,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow), 1U) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i))) 
                                              << 3U)))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we = 0U;
    if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state) 
                  >> 2U)))) {
        if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we)
                         ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i)
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc) 
                            + (0x000007f0U & ((((IData)(0x0000000bU) 
                                                * (
                                                   VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh), 1U) 
                                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i))) 
                                               + (VL_SHIFTL_III(7,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow), 1U) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i))) 
                                              << 4U)))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__a7_dout = 
        ((9U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr))
          ? vlSelfRef.tb_bench__DOT__u_top__DOT__u_a7__DOT__mem
         [vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr]
          : 0U);
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr 
        = (0x00000fffU & (((IData)(0x00000190U) * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i)) 
                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i)));
    if (vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a7__DOT____Vlvbound_hb1bb03c6__0 
            = (VL_LTS_III(32, 0x0000007fU, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out)
                ? 0x0000007fU : (VL_GTS_III(32, 0xffffff80U, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out)
                                  ? 0x00000080U : (0x000000ffU 
                                                   & vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out)));
        if ((9U >= (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr))) {
            __VdlyVal__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__u_a7__DOT____Vlvbound_hb1bb03c6__0;
            __VdlyDim0__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 
                = vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr;
            __VdlySet__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0 = 1U;
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done = 0U;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr = 
        ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we)
          ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i)
          : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done = 0U;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr = 
        (0x000001ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we)
                         ? ((0x000001f0U & ((((IData)(5U) 
                                              * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh)) 
                                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow)) 
                                            << 4U)) 
                            + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done = 0U;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we)
                         ? ((0x000007f8U & ((((IData)(0x0000000dU) 
                                              * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh)) 
                                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow)) 
                                            << 3U)) 
                            + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc))
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic) 
                            + (0x000007f8U & (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow) 
                                               + (((IData)(0x0000000dU) 
                                                   * 
                                                   ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i) 
                                                    + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh))) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i))) 
                                              << 3U)))));
    __Vtableidx1 = (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage) 
                     << 2U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start = Vtb_bench__ConstPool__TABLE_had1126da_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start = Vtb_bench__ConstPool__TABLE_h21db8459_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start = Vtb_bench__ConstPool__TABLE_h0e45ff86_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start = Vtb_bench__ConstPool__TABLE_h392ef149_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start = Vtb_bench__ConstPool__TABLE_h8ddecfd7_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start = Vtb_bench__ConstPool__TABLE_hce4ff95b_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start = Vtb_bench__ConstPool__TABLE_hd55a8f63_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start = Vtb_bench__ConstPool__TABLE_hd25ea8a5_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__done = Vtb_bench__ConstPool__TABLE_hf85a0a8a_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr 
        = (0x0000007fU & (((IData)(3U) * (((IData)(3U) 
                                           * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc)) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i))) 
                          + ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic) 
                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr 
        = (0x000007ffU & ((0x000007f8U & ((((IData)(3U) 
                                            * (((IData)(3U) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i))) 
                                           + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i)) 
                                          << 3U)) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done = 0U;
    tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_2)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_2 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_2), 0x0000001fU)))));
    tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_0)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_0 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_0), 0x0000001fU)))));
    tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_1)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_1 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_1), 0x0000001fU)))));
    if (__VdlySet__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_a7__DOT__mem[__VdlyDim0__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0] 
            = __VdlyVal__tb_bench__DOT__u_top__DOT__u_a7__DOT__mem__v0;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done = 1U;
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 0U;
                }
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))
                ? ((0x151fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done = 1U;
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 0U;
                }
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))
                ? ((0x078fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done = 1U;
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
                        = ((9U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                    ? 8U : ((0x018fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done = 1U;
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 0U;
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
                    = ((((0x0cU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh)) 
                         & (0x0cU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow))) 
                        & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc)))
                        ? 5U : 1U);
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))
                ? (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done = 1U;
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 0U;
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
                    = ((((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh)) 
                         & (4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow))) 
                        & (0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc)))
                        ? 5U : 1U);
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))
                ? (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr = 
        (0x00001fffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc) 
                            + (0x00001ff8U & ((((IData)(0x0000001aU) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow)) 
                                              << 3U)))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i)));
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done = 1U;
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
                        = ((((0x19U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh)) 
                             & (0x19U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow))) 
                            & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc)))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                    ? 8U : ((((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i)) 
                              & (2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))) 
                             & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic)))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc) 
                            + (0x000007f0U & ((((IData)(0x0000000bU) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow)) 
                                              << 4U)))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i)));
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done = 1U;
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
                        = ((((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh)) 
                             & (0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow))) 
                            & (0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc)))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                    ? 8U : ((((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i)) 
                              & (2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i))) 
                             & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic)))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out)))))))))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out)))))))))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out)))))))))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done = 0U;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l7_y_we = 1U;
            }
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 0U;
                }
                vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done = 1U;
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 5U;
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))
                ? ((9U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 1U;
    }
}

void Vtb_bench___024root___eval_nba(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_nba\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vtb_bench___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((7ULL & vlSelfRef.__VnbaTriggered[0U])) {
        Vtb_bench___024root___act_comb__TOP__0(vlSelf);
    }
}

void Vtb_bench___024root___timing_ready(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___timing_ready\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VtrigSched_ha7a70230__0.ready("@(posedge tb_bench.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VtrigSched_h94de5d66__0.ready("@( tb_bench.done)");
    }
}

void Vtb_bench___024root___timing_resume(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___timing_resume\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VtrigSched_ha7a70230__0.moveToResumeQueue(
                                                          "@(posedge tb_bench.clk)");
    vlSelfRef.__VtrigSched_h94de5d66__0.moveToResumeQueue(
                                                          "@( tb_bench.done)");
    vlSelfRef.__VtrigSched_ha7a70230__0.resume("@(posedge tb_bench.clk)");
    vlSelfRef.__VtrigSched_h94de5d66__0.resume("@( tb_bench.done)");
    if ((2ULL & vlSelfRef.__VactTriggered[0U])) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtb_bench___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___trigger_orInto__act_vec_vec\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((IData)(1U) + n);
    } while ((0U >= n));
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_bench___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vtb_bench___024root___eval_phase__act(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_phase__act\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb_bench___024root___eval_triggers_vec__act(vlSelf);
    Vtb_bench___024root___timing_ready(vlSelf);
    Vtb_bench___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_bench___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_bench___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_bench___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
    if (__VactExecute) {
        vlSelfRef.__VactTriggeredAcc.fill(0ULL);
        Vtb_bench___024root___timing_resume(vlSelf);
        Vtb_bench___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_bench___024root___eval_phase__inact(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_phase__inact\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("tb_bench.sv", 4, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_bench___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtb_bench___024root___eval_phase__nba(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_phase__nba\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_bench___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        Vtb_bench___024root___eval_nba(vlSelf);
        Vtb_bench___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtb_bench___024root___eval(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_bench___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("tb_bench.sv", 4, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VinactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VinactIterCount)))) {
                VL_FATAL_MT("tb_bench.sv", 4, "", "DIDNOTCONVERGE: Inactive region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VinactIterCount = ((IData)(1U) 
                                           + vlSelfRef.__VinactIterCount);
            vlSelfRef.__VactIterCount = 0U;
            do {
                if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                    Vtb_bench___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                    VL_FATAL_MT("tb_bench.sv", 4, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
                }
                vlSelfRef.__VactIterCount = ((IData)(1U) 
                                             + vlSelfRef.__VactIterCount);
                vlSelfRef.__VactPhaseResult = Vtb_bench___024root___eval_phase__act(vlSelf);
            } while (vlSelfRef.__VactPhaseResult);
            vlSelfRef.__VinactPhaseResult = Vtb_bench___024root___eval_phase__inact(vlSelf);
        } while (vlSelfRef.__VinactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vtb_bench___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

void Vtb_bench___024root____VbeforeTrig_ha7a70230__0(Vtb_bench___024root* vlSelf, const char* __VeventDescription) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root____VbeforeTrig_ha7a70230__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlUnpacked<QData/*63:0*/, 1> __VTmp;
    // Body
    __VTmp[0U] = (QData)((IData)(((IData)(vlSelfRef.tb_bench__DOT__clk) 
                                  & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0)))));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0 
        = vlSelfRef.tb_bench__DOT__clk;
    if ((1ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_ha7a70230__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_ha7a70230__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_ha7a70230__0.ready(__VeventDescription);
        vlSelfRef.__VtrigSched_ha7a70230__0.ready(__VeventDescription);
    }
    vlSelfRef.__VactTriggeredAcc[0U] = (vlSelfRef.__VactTriggeredAcc[0U] 
                                        | __VTmp[0U]);
}

void Vtb_bench___024root____VbeforeTrig_h94de5d66__0(Vtb_bench___024root* vlSelf, const char* __VeventDescription) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root____VbeforeTrig_h94de5d66__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlUnpacked<QData/*63:0*/, 1> __VTmp;
    // Body
    __VTmp[0U] = (QData)((IData)((((IData)(vlSelfRef.tb_bench__DOT__done) 
                                   != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__done__0)) 
                                  << 2U)));
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__done__0 
        = vlSelfRef.tb_bench__DOT__done;
    if ((4ULL & __VTmp[0U])) {
        vlSelfRef.__VtrigSched_h94de5d66__0.ready(__VeventDescription);
    }
    vlSelfRef.__VactTriggeredAcc[0U] = (vlSelfRef.__VactTriggeredAcc[0U] 
                                        | __VTmp[0U]);
}

#ifdef VL_DEBUG
void Vtb_bench___024root___eval_debug_assertions(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_debug_assertions\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
