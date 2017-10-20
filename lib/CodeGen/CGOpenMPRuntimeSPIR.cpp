//===----- CGOpenMPRuntimeSPIR.h - Interface to OpenMP SPIR Runtimes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to SPIR
// targets.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeSPIR.h"
#include <iostream>

using namespace clang;
using namespace CodeGen;


/// \brief RAII for emitting code of OpenMP constructs.
class OutlinedFunctionRAII {
  CGOpenMPRuntimeSPIR &RT;
  CodeGenModule &CGM;
  llvm::BasicBlock * oldMCB;
  CodeGenFunction * oldCGF;

public:
  OutlinedFunctionRAII(CGOpenMPRuntimeSPIR &RT, CodeGenModule &CGM)
    : RT(RT), CGM(CGM), oldMCB(RT.MasterContBlock), oldCGF(RT.currentCGF) {

    RT.MasterContBlock = nullptr;
    RT.currentCGF = nullptr;
  }
  ~OutlinedFunctionRAII() {
    assert(RT.MasterContBlock == nullptr && "master region was not closed.");
    RT.MasterContBlock = oldMCB;
    RT.currentCGF = oldCGF;
  }

};

CGOpenMPRuntimeSPIR::CGOpenMPRuntimeSPIR(CodeGenModule &CGM)
        : CGOpenMPRuntime(CGM) {
  CGM.ASTAllocaAddressSpace = LangAS::opencl_generic; // this is to ensure that alloca gets casted to Default
  MasterContBlock = nullptr;
  NumThreadsContBlock = nullptr;
  NumThreads = nullptr;
  inParallel = false;
  isTargetParallel = false;
  std::cout << "using SPIR-V\n";
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP SPIR can only handle device code.");
}

llvm::Constant * CGOpenMPRuntimeSPIR::createRuntimeFunction(OpenMPRTLFunctionSPIR Function) {
  llvm::Type *param[] = {CGM.Int32Ty};
  switch (Function) {
    case get_global_id: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, param, false), "_Z13get_global_idj");
    }
    case get_local_id: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, param, false), "_Z12get_local_idj");
    }
    case get_local_size: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, param, false), "_Z14get_local_sizej");
    }
    case get_num_groups: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, param, false), "_Z14get_num_groupsj");
    }
    case get_group_id: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, param, false), "_Z12get_group_idj");
    }
    case work_group_barrier: {
      //CLK_GLOBAL_MEM_FENCE   0x02 or (1 << 1)
      //CLK_LOCAL_MEM_FENCE    0x01 or (1 << 0)
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.VoidTy, param, false), "_Z18work_group_barrierj");
    }
    case mem_fence: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.VoidTy, param, false), "_Z9mem_fencej");
    }
    case read_mem_fence: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.VoidTy, param, false), "_Z14read_mem_fencej");
    }
    case write_mem_fence: {
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.VoidTy, param, false), "_Z15write_mem_fencej");
    }
    default:
      return nullptr;
  }
}

void CGOpenMPRuntimeSPIR::emitMasterHeader(CodeGenFunction &CGF) {
  assert(MasterContBlock == nullptr && "cannot open two master regions");
  llvm::Value *arg[] = {CGF.Builder.getInt32(0)};
  llvm::CallInst *ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value *ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  llvm::Value *cond = CGF.Builder.CreateIsNull(ltid_casted);
  llvm::BasicBlock * ThenBlock = CGF.createBasicBlock("omp_if.then.master");
  MasterContBlock = CGF.createBasicBlock("omp_if.end.master");
  currentCGF = &CGF;
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(cond, ThenBlock, MasterContBlock);
  CGF.EmitBlock(ThenBlock);
  return;
}

void CGOpenMPRuntimeSPIR::emitMasterFooter() {
  // only close master region, if one is open
  if (MasterContBlock == nullptr)
    return;
  currentCGF->EmitBranch(MasterContBlock);
  currentCGF->EmitBlock(MasterContBlock, true);
  MasterContBlock = nullptr;
  return;
}

void CGOpenMPRuntimeSPIR::emitNumThreadsHeader(CodeGenFunction &CGF, llvm::Value *NumThreads) {
  assert(NumThreadsContBlock == nullptr);
  llvm::Value *arg[] = {CGF.Builder.getInt32(0)};
  llvm::CallInst *ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value *ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  llvm::Value *cond = CGF.Builder.CreateICmpSLT(ltid_casted, NumThreads);
  llvm::BasicBlock *ThenBlock = CGF.createBasicBlock("omp_if.then.num_threads");
  NumThreadsContBlock = CGF.createBasicBlock("omp_if.end.num_threads");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(cond, ThenBlock, NumThreadsContBlock);
  CGF.EmitBlock(ThenBlock);
  return;
}

void CGOpenMPRuntimeSPIR::emitNumThreadsFooter(CodeGenFunction &CGF) {
  // only close num_threads region, if there is one
  if (NumThreadsContBlock == nullptr)
    return;
  CGF.EmitBranch(NumThreadsContBlock);
  CGF.EmitBlock(NumThreadsContBlock, true);
  NumThreadsContBlock = nullptr;
  return;
}

bool CGOpenMPRuntimeSPIR::targetHasInnerOutlinedFunction(OpenMPDirectiveKind kind) {
  switch (kind) {
    case OpenMPDirectiveKind::OMPD_target_parallel:
    case OpenMPDirectiveKind::OMPD_target_parallel_for:
    case OpenMPDirectiveKind::OMPD_target_parallel_for_simd:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for_simd:
      isTargetParallel = true;
    case OpenMPDirectiveKind::OMPD_target_teams:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_simd:
      return true;
    default:
      return false;
  }
}

bool CGOpenMPRuntimeSPIR::teamsHasInnerOutlinedFunction(OpenMPDirectiveKind kind) {
  switch (kind) {
    case OpenMPDirectiveKind::OMPD_teams_distribute_parallel_for:
    case OpenMPDirectiveKind::OMPD_teams_distribute_parallel_for_simd:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for_simd:
      isTargetParallel = true;
      return true;
    default:
      return false;
  }
}
/*
static unsigned ArgInfoAddressSpace(unsigned LangAS) {
  switch (LangAS) {
    case LangAS::opencl_global:   return 1;
    case LangAS::opencl_constant: return 2;
    case LangAS::opencl_local:    return 3;
    case LangAS::opencl_generic:  return 4; // Not in SPIR 2.0 specs.
    default:
      return 0; // Assume private.
  }
}*/

// TODO clean up unnecessary code
void CGOpenMPRuntimeSPIR::GenOpenCLArgMetadata(const RecordDecl *FD, llvm::Function *Fn,
                                 CodeGenModule &CGM) {
  CodeGenFunction CGF(CGM);
  llvm::LLVMContext &Context = CGM.getLLVMContext();
  CGBuilderTy Builder = CGF.Builder;
  SmallVector<llvm::Metadata *, 8> opSource = {
          llvm::ConstantAsMetadata::get(Builder.getInt32(3)), // OpenCL C
          llvm::ConstantAsMetadata::get(Builder.getInt32(10000))}; // OpenCL C Version
  llvm::MDNode * srcMD = llvm::MDNode::get(Context, opSource);
  Fn->getParent()->getOrInsertNamedMetadata("spirv.Source")->addOperand(srcMD);
  // Create MDNodes that represent the kernel arg metadata.
  // Each MDNode is a list in the form of "key", N number of values which is
  // the same number of values as their are kernel arguments.
  ASTContext &ASTCtx = CGM.getContext();

  const PrintingPolicy &Policy = ASTCtx.getPrintingPolicy();

  // MDNode for the kernel argument address space qualifiers.
  SmallVector<llvm::Metadata *, 8> addressQuals;

  // MDNode for the kernel argument access qualifiers (images only).
  SmallVector<llvm::Metadata *, 8> accessQuals;

  // MDNode for the kernel argument type names.
  SmallVector<llvm::Metadata *, 8> argTypeNames;

  // MDNode for the kernel argument base type names.
  SmallVector<llvm::Metadata *, 8> argBaseTypeNames;

  // MDNode for the kernel argument type qualifiers.
  SmallVector<llvm::Metadata *, 8> argTypeQuals;

  // MDNode for the kernel argument names.
  SmallVector<llvm::Metadata *, 8> argNames;

  //for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
  //  const ParmVarDecl *parm = FD->getParamDecl(i);
  for(auto parm : FD->fields()) {
    QualType ty = parm->getType();
    std::string typeQuals;

    if (ty->isPointerType()) {
      QualType pointeeTy = ty->getPointeeType();

      // Get address qualifier.
      addressQuals.push_back(llvm::ConstantAsMetadata::get(Builder.getInt32(
              CGM.getContext().getTargetAddressSpace(pointeeTy.getAddressSpace()))));

      // Get argument type name.
      std::string typeName =
              pointeeTy.getUnqualifiedType().getAsString(Policy) + "*";

      // Turn "unsigned type" to "utype"
      std::string::size_type pos = typeName.find("unsigned");
      if (pointeeTy.isCanonical() && pos != std::string::npos)
        typeName.erase(pos + 1, 8);

      argTypeNames.push_back(llvm::MDString::get(Context, typeName));

      std::string baseTypeName =
              pointeeTy.getUnqualifiedType().getCanonicalType().getAsString(
                      Policy) +
              "*";

      // Turn "unsigned type" to "utype"
      pos = baseTypeName.find("unsigned");
      if (pos != std::string::npos)
        baseTypeName.erase(pos + 1, 8);

      argBaseTypeNames.push_back(llvm::MDString::get(Context, baseTypeName));

      // Get argument type qualifiers:
      if (ty.isRestrictQualified())
        typeQuals = "restrict";
      if (pointeeTy.isConstQualified() ||
          (pointeeTy.getAddressSpace() == LangAS::opencl_constant))
        typeQuals += typeQuals.empty() ? "const" : " const";
      if (pointeeTy.isVolatileQualified())
        typeQuals += typeQuals.empty() ? "volatile" : " volatile";
    } else {
      uint32_t AddrSpc = 0;
      bool isPipe = ty->isPipeType();
      if (ty->isImageType() || isPipe)
        AddrSpc = CGM.getContext().getTargetAddressSpace(LangAS::opencl_global);

      addressQuals.push_back(
              llvm::ConstantAsMetadata::get(Builder.getInt32(AddrSpc)));

      // Get argument type name.
      std::string typeName;
      if (isPipe)
        typeName = ty.getCanonicalType()->getAs<PipeType>()->getElementType()
                .getAsString(Policy);
      else
        typeName = ty.getUnqualifiedType().getAsString(Policy);

      // Turn "unsigned type" to "utype"
      std::string::size_type pos = typeName.find("unsigned");
      if (ty.isCanonical() && pos != std::string::npos)
        typeName.erase(pos + 1, 8);

      std::string baseTypeName;
      if (isPipe)
        baseTypeName = ty.getCanonicalType()->getAs<PipeType>()
                ->getElementType().getCanonicalType()
                .getAsString(Policy);
      else
        baseTypeName =
                ty.getUnqualifiedType().getCanonicalType().getAsString(Policy);


      argTypeNames.push_back(llvm::MDString::get(Context, typeName));

      // Turn "unsigned type" to "utype"
      pos = baseTypeName.find("unsigned");
      if (pos != std::string::npos)
        baseTypeName.erase(pos + 1, 8);

      argBaseTypeNames.push_back(llvm::MDString::get(Context, baseTypeName));


      argTypeQuals.push_back(llvm::MDString::get(Context, typeQuals));

      // Get image and pipe access qualifier:
      if (ty->isImageType() || ty->isPipeType()) {
        const OpenCLAccessAttr *A = parm->getAttr<OpenCLAccessAttr>();
        if (A && A->isWriteOnly())
          accessQuals.push_back(llvm::MDString::get(Context, "write_only"));
        else if (A && A->isReadWrite())
          accessQuals.push_back(llvm::MDString::get(Context, "read_write"));
        else
          accessQuals.push_back(llvm::MDString::get(Context, "read_only"));
      } else
        accessQuals.push_back(llvm::MDString::get(Context, "none"));

      // Get argument name.
      argNames.push_back(llvm::MDString::get(Context, parm->getName()));
    }

    Fn->setMetadata("kernel_arg_addr_space",
                    llvm::MDNode::get(Context, addressQuals));
    Fn->setMetadata("kernel_arg_access_qual",
                    llvm::MDNode::get(Context, accessQuals));
    Fn->setMetadata("kernel_arg_type",
                    llvm::MDNode::get(Context, argTypeNames));
    Fn->setMetadata("kernel_arg_base_type",
                    llvm::MDNode::get(Context, argBaseTypeNames));
    Fn->setMetadata("kernel_arg_type_qual",
                    llvm::MDNode::get(Context, argTypeQuals));
    if (CGM.getCodeGenOpts().EmitOpenCLArgMetadata)
      Fn->setMetadata("kernel_arg_name",
                      llvm::MDNode::get(Context, argNames));
  }
}

void CGOpenMPRuntimeSPIR::emitTargetOutlinedFunction(
        const OMPExecutableDirective &D, StringRef ParentName,
        llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
        bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");
  //std::cout << "emit target outlined function\n";
  //setTargetParallel(D.getDirectiveKind());
  CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());
  for (auto capture : CS.captures()) {
    globals.insert(capture.getCapturedVar()->getDeclName());
  }

  OutlinedFunctionRAII RAII(*this, CGM);
  class MasterPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeSPIR &RT;
  public:
    MasterPrePostActionTy(CGOpenMPRuntimeSPIR &RT) : RT(RT) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitMasterHeader(CGF);
    }

    void Exit(CodeGenFunction &CGF) override {
      RT.emitMasterFooter();
    }
  } Action(*this);
  if (!targetHasInnerOutlinedFunction(D.getDirectiveKind())) {
    CodeGen.setAction(Action);
  }
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
  //std::cout << "Target outlined function emitted\n";
  OutlinedFn->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  OutlinedFn->addFnAttr(llvm::Attribute::NoUnwind);
  OutlinedFn->removeFnAttr(llvm::Attribute::OptimizeNone);
  //OutlinedFn->dump();
  GenOpenCLArgMetadata(CS.getCapturedRecordDecl(), OutlinedFn, CGM);
}

llvm::Value *CGOpenMPRuntimeSPIR::emitParallelOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  // first, we must close any open master sections:
  // TODO: only in "distribute parallel for" directives
  //if (isOpenMPDistributeDirective(D.getDirectiveKind())) {
    this->emitMasterFooter();
  //}

  // TODO: create PostActionTy to broadcast variable from thread lastIt to all in WG, then leave it private?
  llvm::DenseSet<const VarDecl *> Lastprivates;
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    for (const auto *D : C->varlists())
      Lastprivates.insert(cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl())->getCanonicalDecl());
  }

  llvm::DenseSet<DeclarationName> FirstPrivates;
  // TODO: create PreActionTy to broadcast variable from thread 0 to all in WG?
  for (const auto *C : D.getClausesOfKind<OMPFirstprivateClause>()) {
    for (const auto *D : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl());
      bool ThisFirstprivateIsLastprivate =
              Lastprivates.count(OrigVD->getCanonicalDecl()) > 0;
      if (!ThisFirstprivateIsLastprivate)
        FirstPrivates.insert(OrigVD->getDeclName());
    }
  }

  const CapturedStmt *CS = D.getCapturedStmt(OMPD_parallel);
  int i = 0;
  isShared.resize(CS->capture_size());
  //std::cout << "number of captures: " << CS->capture_size() << "\n";
  for (auto capture : CS->captures()) {
    DeclarationName name = capture.getCapturedVar()->getDeclName();
    if (globals.count(name) + FirstPrivates.count(name)  == 0) { // not global, not private
      isShared.set(i);
    }
    ++i;
  }

  //std::cout << "emit parallel outlined function\n";
  bool wasAlreadyParallel = inParallel;
  inParallel = true;
  OutlinedFunctionRAII RAII(*this, CGM);
  llvm::Value *OutlinedFn = CGOpenMPRuntime::emitParallelOutlinedFunction(D, ThreadIDVar, InnermostKind, CodeGen);
  inParallel = wasAlreadyParallel;
  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
    Fn->addFnAttr(llvm::Attribute::NoUnwind);
  }
  //OutlinedFn->dump();
  return OutlinedFn;
}

void CGOpenMPRuntimeSPIR::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars,
                                       const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::SmallVector<llvm::Value *, 16> RealArgs;
  RealArgs.push_back(llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  RealArgs.push_back(llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  RealArgs.append(CapturedVars.begin(), CapturedVars.end());

  llvm::Function * F = cast<llvm::Function> (OutlinedFn);
  //F->getFunctionType()->dump();

  //std::cout << "number of params: " << F->getFunctionType()->getNumParams() << "\n";

  //if(inParallel) {
    // we are either in a nested parallel region or already have initialization code from distribute directive
    // so, we predicate the following copy instructions
    emitMasterHeader(CGF);
  //}
  llvm::DenseMap<llvm::Value *, llvm::GlobalVariable *> sharedVariables;
  bool emitBarrier = false;
  unsigned addrSpaceLocal = CGM.getContext().getTargetAddressSpace(LangAS::opencl_local);
  unsigned addrSpaceGeneric = CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic);
  unsigned skip = F->getFunctionType()->getNumParams() - (unsigned) isShared.size();

  for (unsigned i = skip; i < F->getFunctionType()->getNumParams(); ++i) { // emit global_tid and bound_tid ...
    if (isShared.test(i-skip)) {
      // copy to scratchpad memory:
      // Create Global Variable with name <function name>.<variable name>
      // store arguments value in global variable and
      // replace argument by pointer to global variable (casted to generic addrspace)
      llvm::PointerType * argType = cast<llvm::PointerType>(RealArgs[i]->getType());
      llvm::Value * arg = CGF.Builder.CreateAlignedLoad(RealArgs[i], CGM.getDataLayout().getPrefTypeAlignment(argType->getElementType()));
      const Twine &name = Twine(F->getName()) + Twine(".") + Twine(RealArgs[i]->getName());
      llvm::GlobalVariable * sharedVarPtr = new llvm::GlobalVariable(CGM.getModule(), arg->getType(), false, llvm::GlobalVariable::InternalLinkage,
                                                                  llvm::Constant::getNullValue(arg->getType()), name, nullptr,
                                                                  llvm::GlobalVariable::NotThreadLocal, addrSpaceLocal);
      CGF.Builder.CreateAlignedStore(arg, sharedVarPtr, CGM.getDataLayout().getPrefTypeAlignment(arg->getType()));
      llvm::Value * newArg = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              sharedVarPtr, sharedVarPtr->getType()->getPointerElementType()->getPointerTo(addrSpaceGeneric));
      emitBarrier = true;
      sharedVariables[RealArgs[i]] = sharedVarPtr;
      RealArgs[i] = newArg;
    }
  }
  isShared.clear();
  emitMasterFooter();
  // memory fence to wait for stores to local mem:
  if (emitBarrier) {
    // call opencl write_mem_fence
    llvm::Value * arg[] = { CGF.Builder.getInt32(1 << 0) }; //CLK_LOCAL_MEM_FENCE   0x01
    CGF.EmitRuntimeCall(createRuntimeFunction(write_mem_fence), arg);
  }
  for (llvm::Value * arg : RealArgs) {
    //arg->dump();
  }
  // call outlined parallel function:
  CGF.EmitCallOrInvoke(OutlinedFn, RealArgs);

  if(isTargetParallel)
    return;

  // copy back shared variables to threadlocal
  emitMasterHeader(CGF);
  for (auto pair : sharedVariables) {
    llvm::PointerType *argType = cast<llvm::PointerType>(pair.first->getType());
    llvm::Value *sharedVar = CGF.Builder.CreateAlignedLoad(pair.second, CGM.getDataLayout().getPrefTypeAlignment(
            argType->getElementType()));
    CGF.Builder.CreateAlignedStore(sharedVar, pair.first,
                                   CGM.getDataLayout().getPrefTypeAlignment(sharedVar->getType()));
  }
  if (emitBarrier) {
    // call opencl read_mem_fence
    llvm::Value * arg[] = { CGF.Builder.getInt32(1 << 0) }; //CLK_LOCAL_MEM_FENCE   0x01
    CGF.EmitRuntimeCall(createRuntimeFunction(read_mem_fence), arg);
  }
  if (inParallel) {
    emitMasterFooter();
  } else if (NumThreads) {
    emitNumThreadsFooter(CGF);
    NumThreads = nullptr;
  }
}


llvm::Value *CGOpenMPRuntimeSPIR::emitTeamsOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  //std::cout << "emit teams outlined funtion\n";

  OutlinedFunctionRAII RAII(*this, CGM);
  class TeamsPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeSPIR &RT;
  public:
    TeamsPrePostActionTy(CGOpenMPRuntimeSPIR &RT)
            : RT(RT) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitMasterHeader(CGF);
    }
    void Exit(CodeGenFunction &CGF) override {
      RT.emitMasterFooter();
    }
  } Action(*this);
  if (!teamsHasInnerOutlinedFunction(D.getDirectiveKind()))
    CodeGen.setAction(Action);

  llvm::Value *OutlinedFn = CGOpenMPRuntime::emitTeamsOutlinedFunction(D, ThreadIDVar, InnermostKind, CodeGen);
  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  //OutlinedFn->dump();
  return OutlinedFn;

}

void CGOpenMPRuntimeSPIR::emitTeamsCall(CodeGenFunction &CGF,
                                        const OMPExecutableDirective &D,
                                        SourceLocation Loc,
                                        llvm::Value *OutlinedFn,
                                        ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  emitMasterFooter();
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  OutlinedFnArgs.push_back(llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}

void CGOpenMPRuntimeSPIR::emitMasterRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &MasterOpGen,
                                       SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  // principle: if(threadID == 0):
  emitMasterHeader(CGF);
  emitInlinedDirective(CGF, OMPD_master, MasterOpGen);
  emitMasterFooter();
}

void CGOpenMPRuntimeSPIR::emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                      OpenMPDirectiveKind Kind, bool EmitChecks,
                                      bool ForceSimpleCall) {
  if (!CGF.HaveInsertPoint())
    return;

  // call opencl work group barrier
  llvm::Value * arg[] = { CGF.Builder.getInt32(1 << 1) }; //CLK_GLOBAL_MEM_FENCE   0x02
  CGF.EmitRuntimeCall(createRuntimeFunction(work_group_barrier), arg);
}

void CGOpenMPRuntimeSPIR::emitForStaticInit(CodeGenFunction &CGF, SourceLocation Loc,
                       OpenMPDirectiveKind DKind,
                       const OpenMPScheduleTy &ScheduleKind,
                       const CGOpenMPRuntime::StaticRTInput &Values) {

  LValue LBLValue = CGF.MakeAddrLValue(Values.LB, CGF.getContext().getIntPtrType());
  LValue UBLValue = CGF.MakeAddrLValue(Values.UB, CGF.getContext().getIntPtrType());
  LValue STLValue = CGF.MakeAddrLValue(Values.ST, CGF.getContext().getIntPtrType());
  LValue ILLValue = CGF.MakeAddrLValue(Values.IL, CGF.getContext().getIntPtrType());
  // take lb
  llvm::Value * lb = CGF.EmitLoadOfScalar(LBLValue, Loc);
  // get int type from values:
  llvm::Type * intType = CGF.Builder.getIntNTy(Values.IVSize);//lb->getType();
  //std::cout << "intType is: " << intType->getPrimitiveSizeInBits() << ", while size says: " << Values.IVSize << "\n";
  llvm::Value * one = CGF.Builder.getIntN(Values.IVSize /*intType->getPrimitiveSizeInBits()*/,1);

  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst *local_size = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_size), arg);
  llvm::Value * NumThreads = CGF.Builder.CreateSExtOrTrunc(local_size, intType);
  llvm::Value *ub = CGF.EmitLoadOfScalar(UBLValue, Loc);
  llvm::Value *it_space = CGF.Builder.CreateSub(ub, lb);

  llvm::Value * chunk = Values.Chunk;
  if (chunk == nullptr) {
    /** Static Scheduling:
     * When no chunk_size is specified, the iteration space is divided into chunks
     * that are approximately equal in size, and at most one chunk is distributed to
     * each thread. The size of the chunks is unspecified in this case.
     */
    if (ScheduleKind.Schedule == OpenMPScheduleClauseKind::OMPC_SCHEDULE_static) {
      // here we do: chunksize = (ub-lb+local_size)/local_size
      llvm::Value *it_space_rounded = CGF.Builder.CreateAdd(it_space, NumThreads);
      chunk = CGF.Builder.CreateSDiv(it_space_rounded, NumThreads);
    } else {
      // If no schedule is specified, scheduling is implementation defined.
      // For the spir target, we choose schedule(static,1) as default
      chunk = one;
    }
  }
  llvm::CallInst * ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value * ltid_casted = CGF.Builder.CreateSExtOrTrunc(ltid, intType);

  // is last is: (it_space/chunk % local_size) == local_id
  llvm::Value * thread_iters = CGF.Builder.CreateSDiv(it_space, chunk);
  llvm::Value * last_iter = CGF.Builder.CreateURem(thread_iters, NumThreads);
  llvm::Value * last_iter_flag = CGF.Builder.CreateICmpEQ(last_iter, ltid_casted);
  llvm::Value * last_iter_flag_casted = CGF.Builder.CreateSExtOrTrunc(last_iter_flag, CGF.Builder.getInt32Ty());
  CGF.EmitStoreOfScalar(last_iter_flag_casted, ILLValue, true);

  // lower bound is: provided lb + localthreadID * chunksize
  llvm::Value * lbdiff = CGF.Builder.CreateMul(ltid_casted, chunk);
  lb = CGF.Builder.CreateAdd(lb, lbdiff);
  CGF.EmitStoreOfScalar(lb, LBLValue, true);

  // upper bound is: lb + chunk-1 (for chunksize=1, this results in lb=ub)
  llvm::Value * ch = CGF.Builder.CreateSub(chunk, one);
  ub = CGF.Builder.CreateAdd(lb, ch);
  CGF.EmitStoreOfScalar(ub, UBLValue, true);

  // stride is: local workgroup size * chunksize
  llvm::Value * stride = CGF.Builder.CreateMul(NumThreads, chunk);
  CGF.EmitStoreOfScalar(stride, STLValue, true);
  return;
}

void CGOpenMPRuntimeSPIR::emitDistributeStaticInit(CodeGenFunction &CGF,
                              SourceLocation Loc,
                              OpenMPDistScheduleClauseKind SchedKind,
                              const CGOpenMPRuntime::StaticRTInput &Values) {
  LValue LBLValue = CGF.MakeAddrLValue(Values.LB, CGF.getContext().getIntPtrType());
  LValue UBLValue = CGF.MakeAddrLValue(Values.UB, CGF.getContext().getIntPtrType());
  LValue STLValue = CGF.MakeAddrLValue(Values.ST, CGF.getContext().getIntPtrType());
  LValue ILLValue = CGF.MakeAddrLValue(Values.IL, CGF.getContext().getIntPtrType());
  // take lb
  llvm::Value * lb = CGF.EmitLoadOfScalar(LBLValue, Loc);
  // get int type from values:
  llvm::Type * intType = lb->getType();
  llvm::Value * one = CGF.Builder.getIntN(intType->getPrimitiveSizeInBits(),1);

  // take num_groups
  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * num_groups = CGF.EmitRuntimeCall(createRuntimeFunction(get_num_groups), arg);
  llvm::Value * num_groups_casted = CGF.Builder.CreateSExtOrTrunc(num_groups, intType);
  llvm::Value *ub = CGF.EmitLoadOfScalar(UBLValue, Loc);
  llvm::Value *it_space = CGF.Builder.CreateSub(ub, lb);

  llvm::Value * chunk = Values.Chunk;
  if (chunk == nullptr) {
    // chunksize is unspecified: calculate a reasonable chunksize
    // chunksize should be multiple of local_size:

    if (SchedKind == OpenMPDistScheduleClauseKind::OMPC_DIST_SCHEDULE_static) {
      /* 2.10.8 distribute construct - dist_schedule(static)
       * When no chunk_size is specified, the iteration space is divided
       * into chunks that are approximately equal in size,
       * and at most one chunk is distributed to each team of the league.
       */
      // here we do: chunksize = ((((ub-lb+local_size)/local_size)+num_groups-1)/num_groups)*local_size
      /*
      llvm::Value * it_space_rounded = CGF.Builder.CreateAdd(it_space, local_size_casted);
      llvm::Value *num_blocks = CGF.Builder.CreateSDiv(it_space_rounded, local_size_casted);
      num_blocks = CGF.Builder.CreateAdd(num_blocks, num_groups_casted);
      num_blocks = CGF.Builder.CreateSub(num_blocks, one);
      chunk = CGF.Builder.CreateMul(CGF.Builder.CreateSDiv(num_blocks, num_groups_casted), local_size_casted);
       */
      llvm::Value * it_space_rounded = CGF.Builder.CreateAdd(it_space, num_groups_casted);
      chunk = CGF.Builder.CreateSDiv(it_space_rounded, num_groups_casted);

    } else {
      // if not static, scheduling is implementation defined:
      // we just assign local_size as chunksize and do round-robin
      llvm::CallInst *local_size = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_size), arg);
      llvm::Value *local_size_casted = CGF.Builder.CreateSExtOrTrunc(local_size, intType);
      chunk = local_size_casted;
    }
  } else { /* chunk already specified */ }

  llvm::CallInst * gid = CGF.EmitRuntimeCall(createRuntimeFunction(get_group_id), arg);
  llvm::Value * gid_casted = CGF.Builder.CreateSExtOrTrunc(gid, intType);

  // is last is: (it_space/chunk % num_groups) == group_id
  llvm::Value * group_iters = CGF.Builder.CreateSDiv(it_space, chunk);
  llvm::Value * last_iter = CGF.Builder.CreateURem(group_iters, num_groups_casted);
  llvm::Value * last_iter_flag = CGF.Builder.CreateICmpEQ(last_iter, gid_casted);
  llvm::Value * last_iter_flag_casted = CGF.Builder.CreateSExtOrTrunc(last_iter_flag, CGF.Builder.getInt32Ty());
  CGF.EmitStoreOfScalar(last_iter_flag_casted, ILLValue, true);

  // lower bound is: lb + groupID * chunksize
  lb = CGF.Builder.CreateAdd(CGF.Builder.CreateMul(gid_casted, chunk), lb);
  CGF.EmitStoreOfScalar(lb, LBLValue, true);

  // upper bound is: lb + chunksize-1
  ub = CGF.Builder.CreateAdd(lb, CGF.Builder.CreateSub(chunk, one));
  CGF.EmitStoreOfScalar(ub, UBLValue, true);

  // stride is: chunksize * num_groups
  llvm::Value * stride = CGF.Builder.CreateMul(chunk, num_groups_casted);
  CGF.EmitStoreOfScalar(stride, STLValue, true);
}

void CGOpenMPRuntimeSPIR::emitForStaticFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                              OpenMPDirectiveKind DKind) {}

void CGOpenMPRuntimeSPIR::emitNumThreadsClause(CodeGenFunction &CGF,
                                           llvm::Value *NumThreads,
                                           SourceLocation Loc) {
  // only emit this clause if it is the outermost parallel construct
  if (inParallel)
    return;
  // principle: if(thread_id < NumThreads) {...}
  //emitNumThreadsHeader(CGF, NumThreads);
  // TODO: put this in bound.tid and use it for for_static_init
  this->NumThreads = NumThreads;
  // Footer must be emitted by end of parallel region
}

void CGOpenMPRuntimeSPIR::emitNumTeamsClause(CodeGenFunction &CGF,
                                         const Expr *NumTeams,
                                         const Expr *ThreadLimit,
                                         SourceLocation Loc) {}

void CGOpenMPRuntimeSPIR::emitForDispatchInit(
        CodeGenFunction &CGF, SourceLocation Loc,
        const OpenMPScheduleTy &ScheduleKind, unsigned IVSize, bool IVSigned,
        bool Ordered, const DispatchRTInput &DispatchValues) {
  llvm_unreachable("For SPIR target, dynamic dispatch is not supported.");
}

bool CGOpenMPRuntimeSPIR::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  // In case of OMPC_SCHEDULE_unknown we return false
  // as we want to emit schedule(static,1) if no schedule clause is specified
  // more precise: the case below is the only one, for which we partition the iteration space
  // into chunks of equal size only to be conformant with the specification
  return (ScheduleKind == OpenMPScheduleClauseKind::OMPC_SCHEDULE_static && !Chunked);
}

bool CGOpenMPRuntimeSPIR::isStaticNonchunked(
        OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  return ScheduleKind == OpenMPDistScheduleClauseKind::OMPC_DIST_SCHEDULE_static && !Chunked;
}

bool CGOpenMPRuntimeSPIR::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  // we don't support real dynamic scheduling and just emit everything as static
  return false;
}

void CGOpenMPRuntimeSPIR::emitInlinedDirective(CodeGenFunction &CGF,
                                           OpenMPDirectiveKind InnerKind,
                                           const RegionCodeGenTy &CodeGen,
                                           bool HasCancel) {
  if (!CGF.HaveInsertPoint())
    return;
  bool oldInParallel = inParallel; // should always be false?!
  switch (InnerKind) {
    case OpenMPDirectiveKind::OMPD_distribute_parallel_for:
    case OpenMPDirectiveKind::OMPD_distribute_parallel_for_simd:
    case OpenMPDirectiveKind::OMPD_distribute:
      inParallel = true;
      emitMasterFooter();
      CGOpenMPRuntime::emitInlinedDirective(CGF, InnerKind, CodeGen, HasCancel);
      inParallel = oldInParallel;
      if(!inParallel)
        emitMasterHeader(CGF);
      break;
    default:
    //std::cout << "unknown Inlined directive!" << "\n";
      CGOpenMPRuntime::emitInlinedDirective(CGF, InnerKind, CodeGen, HasCancel);
  }
  return;
}

void CGOpenMPRuntimeSPIR::createOffloadEntry(llvm::Constant *ID,
                                         llvm::Constant *Addr, uint64_t Size,
                                         int32_t Flags) {}