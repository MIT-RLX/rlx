// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_bench__pch.h"

//============================================================
// Constructors

Vtb_bench::Vtb_bench(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_bench__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_bench::Vtb_bench(const char* _vcname__)
    : Vtb_bench(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_bench::~Vtb_bench() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_bench___024root___eval_debug_assertions(Vtb_bench___024root* vlSelf);
#endif  // VL_DEBUG
void Vtb_bench___024root___eval_static(Vtb_bench___024root* vlSelf);
void Vtb_bench___024root___eval_initial(Vtb_bench___024root* vlSelf);
void Vtb_bench___024root___eval_settle(Vtb_bench___024root* vlSelf);
void Vtb_bench___024root___eval(Vtb_bench___024root* vlSelf);

void Vtb_bench::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_bench::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_bench___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtb_bench___024root___eval_static(&(vlSymsp->TOP));
        Vtb_bench___024root___eval_initial(&(vlSymsp->TOP));
        Vtb_bench___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtb_bench___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtb_bench::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty() && !contextp()->gotFinish(); }

uint64_t Vtb_bench::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_bench::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtb_bench___024root___eval_final(Vtb_bench___024root* vlSelf);

VL_ATTR_COLD void Vtb_bench::final() {
    contextp()->executingFinal(true);
    Vtb_bench___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_bench::hierName() const { return vlSymsp->name(); }
const char* Vtb_bench::modelName() const { return "Vtb_bench"; }
unsigned Vtb_bench::threads() const { return 1; }
void Vtb_bench::prepareClone() const { contextp()->prepareClone(); }
void Vtb_bench::atClone() const {
    contextp()->threadPoolpOnClone();
}
