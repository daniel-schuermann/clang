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

CGOpenMPRuntimeSPIR::CGOpenMPRuntimeSPIR(CodeGenModule &CGM)
        : CGOpenMPRuntime(CGM) {
  MasterContBlock = nullptr;
  NumThreadsContBlock = nullptr;
  inParallel = false;
  std::cout << "using SPIR\n";
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP SPIR can only handle device code.");
  std::cout << std::string(CGM.getDataLayout().getStringRepresentation()) << "\n";
}

llvm::Constant * CGOpenMPRuntimeSPIR::createRuntimeFunction(OpenMPRTLFunctionSPIR Function) {
  llvm::Type *param[] = {CGM.Int32Ty};
  switch(Function) {
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
      //CLK_GLOBAL_MEM_FENCE   0x02
      return CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.VoidTy, param, false), "_Z18work_group_barrierj");
    }
    default:
      return nullptr;
  }
}

// Get the same Type in a given Address Space
QualType CGOpenMPRuntimeSPIR::getAddrSpaceType(QualType T, LangAS::ID AddrSpace) {
  if(T.getTypePtr()->isLValueReferenceType())
    return CGM.getContext().getLValueReferenceType(getAddrSpaceType(T.getTypePtr()->getPointeeType(), AddrSpace), true);
  if(T.getTypePtr()->isAnyPointerType())
    return CGM.getContext().getPointerType(getAddrSpaceType(T.getTypePtr()->getPointeeType(), AddrSpace));
  if(T.getTypePtr()->isBuiltinType())
    return CGM.getContext().getAddrSpaceQualType(T, AddrSpace);
  return T;
}

bool CGOpenMPRuntimeSPIR::isGlobal(IdentifierInfo * info) {
  for(IdentifierInfo * capture : captures) {
    if(capture == info) return true;
  }
  return false;
}

void CGOpenMPRuntimeSPIR::emitMasterHeader(CodeGenFunction &CGF) {
  assert(MasterContBlock == nullptr && "cannot open two master regions");
  llvm::Value *arg[] = {CGF.Builder.getInt32(0)};
  llvm::CallInst *ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value *ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  llvm::Value *cond = CGF.Builder.CreateIsNull(ltid_casted);
  llvm::BasicBlock *ThenBlock = CGF.createBasicBlock("omp_if.then");
  MasterContBlock = CGF.createBasicBlock("omp_if.end");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(cond, ThenBlock, MasterContBlock);
  CGF.EmitBlock(ThenBlock);
  return;
}

void CGOpenMPRuntimeSPIR::emitMasterFooter(CodeGenFunction &CGF) {
  // only close master region, if one is open
  if(MasterContBlock == nullptr)
    return;
  CGF.EmitBranch(MasterContBlock);
  CGF.EmitBlock(MasterContBlock, true);
  MasterContBlock = nullptr;
  return;
}

void CGOpenMPRuntimeSPIR::emitNumThreadsHeader(CodeGenFunction &CGF, llvm::Value *NumThreads) {
  assert(NumThreadsContBlock == nullptr);
  llvm::Value *arg[] = {CGF.Builder.getInt32(0)};
  llvm::CallInst *ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value *ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  llvm::Value *cond = CGF.Builder.CreateICmpSLT(ltid_casted, NumThreads);
  llvm::BasicBlock *ThenBlock = CGF.createBasicBlock("omp_if.then");
  NumThreadsContBlock = CGF.createBasicBlock("omp_if.end");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(cond, ThenBlock, NumThreadsContBlock);
  CGF.EmitBlock(ThenBlock);
  return;
}

void CGOpenMPRuntimeSPIR::emitNumThreadsFooter(CodeGenFunction &CGF) {
  // only close num_threads region, if there is one
  if(NumThreadsContBlock == nullptr)
    return;
  CGF.EmitBranch(NumThreadsContBlock);
  CGF.EmitBlock(NumThreadsContBlock, true);
  NumThreadsContBlock = nullptr;
  return;
}

void CGOpenMPRuntimeSPIR::setTargetParallel(OpenMPDirectiveKind kind) {
  switch(kind) {
    case OpenMPDirectiveKind::OMPD_target_parallel:
    case OpenMPDirectiveKind::OMPD_target_parallel_for:
    case OpenMPDirectiveKind::OMPD_target_parallel_for_simd:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for:
    case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for_simd:
      isTargetParallel = true;
      break;
    default:
      isTargetParallel = false;
  }
  return;
}

static unsigned ArgInfoAddressSpace(unsigned LangAS) {
  switch (LangAS) {
    case LangAS::opencl_global:   return 1;
    case LangAS::opencl_constant: return 2;
    case LangAS::opencl_local:    return 3;
    case LangAS::opencl_generic:  return 4; // Not in SPIR 2.0 specs.
    default:
      return 0; // Assume private.
  }
}

// TODO clean up unnecessary code
void CGOpenMPRuntimeSPIR::GenOpenCLArgMetadata(const RecordDecl *FD, llvm::Function *Fn,
                                 /*CodeGenModule &CGM,*/ llvm::LLVMContext &Context,
                                 CGBuilderTy &Builder) {

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
              ArgInfoAddressSpace(pointeeTy.getAddressSpace()))));

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
        AddrSpc = ArgInfoAddressSpace(LangAS::opencl_global);

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
  std::cout << "emit target outlined function\n";
  setTargetParallel(D.getDirectiveKind());
  CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  const RecordDecl *RD = CS.getCapturedRecordDecl();
  auto I = CS.captures().begin();
    for (auto *FD : RD->fields()) {
      // TODO: are we sure to always have a Pointer here?
      QualType ArgType = FD->getType();
      FD->setType(getAddrSpaceType(ArgType, LangAS::opencl_global));
      captures.push_back(I->getCapturedVar()->getIdentifier());
      ++I;
    }

  class MasterPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeSPIR &RT;
    bool isTargetParallel;
  public:
      MasterPrePostActionTy(CGOpenMPRuntimeSPIR &RT, bool isTargetParallel)
              : RT(RT), isTargetParallel(isTargetParallel) {}
      void Enter(CodeGenFunction &CGF) override {
        if(!isTargetParallel)
          RT.emitMasterHeader(CGF);
      }
      void Exit(CodeGenFunction &CGF) override {
        RT.emitMasterFooter(CGF);
      }
  } Action(*this, isTargetParallel);
  //if(!isTargetParallel)
    CodeGen.setAction(Action);

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);

  OutlinedFn->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  OutlinedFn->addFnAttr(llvm::Attribute::NoUnwind);
  OutlinedFn->removeFnAttr(llvm::Attribute::OptimizeNone);

  CodeGenFunction CGF(CGM);
  GenOpenCLArgMetadata(CS.getCapturedRecordDecl(), OutlinedFn,
                          OutlinedFn->getContext(), CGF.Builder);

}

llvm::Value *CGOpenMPRuntimeSPIR::emitParallelOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_parallel);

  const RecordDecl *RD = CS->getCapturedRecordDecl();
  auto I = CS->captures().begin();
  for (FieldDecl *FD : RD->fields()) {
    if(isGlobal(I->getCapturedVar()->getIdentifier())) {
      FD->setType(getAddrSpaceType(FD->getType(), LangAS::opencl_global));
    }
    ++I;
  }
  std::cout << "emit parallel outlined function\n";
  bool wasAlreadyParallel = inParallel;
  inParallel = true;
  llvm::Value *OutlinedFn = CGOpenMPRuntime::emitParallelOutlinedFunction(D, ThreadIDVar, InnermostKind, CodeGen);
  inParallel = wasAlreadyParallel;
  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  return OutlinedFn;
}

void CGOpenMPRuntimeSPIR::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars,
                                       const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  // TODO: clean up this mess :)
  if(inParallel) {

    std::cout << "call inner parallel function\n";

    auto &&ThenGen = [OutlinedFn, CapturedVars, this](CodeGenFunction &CGF,
                                                       PrePostActionTy &) {
        llvm::SmallVector<llvm::Value *, 16> RealArgs;
        RealArgs.push_back(
                llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo()));
        RealArgs.push_back(
                llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo()));
        RealArgs.append(CapturedVars.begin(), CapturedVars.end());

        CGF.EmitCallOrInvoke(OutlinedFn, RealArgs);
        if(this->NumThreadsContBlock)
          this->emitNumThreadsFooter(CGF);
    };
    RegionCodeGenTy ThenRCG(ThenGen);
    ThenRCG(CGF);

  }else {
    auto &&ThenGen = [OutlinedFn, CapturedVars, this](CodeGenFunction &CGF,
                                                      PrePostActionTy &) {
    emitMasterFooter(CGF);
    // TODO: Better remove these unnecessary arguments?
    llvm::Value *arg[] = {CGF.Builder.getInt32(0)};
    llvm::CallInst *gtid = CGF.EmitRuntimeCall(this->createRuntimeFunction(get_global_id), arg);
    Address global_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".gtid");
    llvm::Value *gtid_casted = CGF.Builder.CreateTruncOrBitCast(gtid, CGF.Int32Ty);
    CGF.EmitStoreOfScalar(gtid_casted, CGF.MakeAddrLValue(global_tid, CGF.getContext().getIntPtrType()), true);

    llvm::CallInst *ltid = CGF.EmitRuntimeCall(this->createRuntimeFunction(get_local_id), arg);
    Address local_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".btid");
    llvm::Value *ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
    CGF.EmitStoreOfScalar(ltid_casted, CGF.MakeAddrLValue(local_tid, CGF.getContext().getIntPtrType()), true);

        llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;

        OutlinedFnArgs.push_back(global_tid.getPointer()); // global_tid
    OutlinedFnArgs.push_back(local_tid.getPointer());  // bound_tid
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());

  for(auto arg : OutlinedFnArgs) arg->dump();
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);

  if(this->NumThreadsContBlock)
    this->emitNumThreadsFooter(CGF); // close num_threads clause, if there is one
  if(!this->isTargetParallel && !this->inParallel) // TODO: we could leave that in and let the optimizer do this for us
    this->emitMasterHeader(CGF);
    };
    RegionCodeGenTy ThenRCG(ThenGen);
    ThenRCG(CGF);
  }
}


llvm::Value *CGOpenMPRuntimeSPIR::emitTeamsOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {

  const CapturedStmt *CS = D.getCapturedStmt(OMPD_teams);
  const RecordDecl *RD = CS->getCapturedRecordDecl();
  for (auto *FD : RD->fields()) {
    FD->setType(getAddrSpaceType(FD->getType(), LangAS::opencl_global));
  }

  return CGOpenMPRuntime::emitTeamsOutlinedFunction(D, ThreadIDVar, InnermostKind, CodeGen);

}

void CGOpenMPRuntimeSPIR::emitTeamsCall(CodeGenFunction &CGF,
                                        const OMPExecutableDirective &D,
                                        SourceLocation Loc,
                                        llvm::Value *OutlinedFn,
                                        ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }

  // TODO: again, remove these arguments?
  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * gtid = CGF.EmitRuntimeCall(createRuntimeFunction(get_global_id), arg);
  Address global_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".gtid");
  llvm::Value * gtid_casted = CGF.Builder.CreateTruncOrBitCast(gtid, CGF.Int32Ty);
  CGF.EmitStoreOfScalar(gtid_casted, CGF.MakeAddrLValue(global_tid, CGF.getContext().getIntPtrType()), true);

  llvm::CallInst * ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  Address local_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".btid");
  llvm::Value * ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  CGF.EmitStoreOfScalar(ltid_casted, CGF.MakeAddrLValue(local_tid, CGF.getContext().getIntPtrType()), true);
  Address ZeroAddr =
          CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                  /*Name*/ ".zero.addr");

  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
  ZeroAddr.getType()->dump();
  for(auto var : CapturedVars) {
    var->dump();
  }
  OutlinedFn->getType()->dump();
  global_tid.getType()->dump();
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(ZeroAddr.getPointer()); // TODO: global_tid
  OutlinedFnArgs.push_back(ZeroAddr.getPointer()); // TODO: bound_tid
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
  emitMasterFooter(CGF);
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

void CGOpenMPRuntimeSPIR::emitForStaticInit(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        const OpenMPScheduleTy &ScheduleKind,
                                        unsigned IVSize, bool IVSigned,
                                        bool Ordered, Address IL, Address LB,
                                        Address UB, Address ST,
                                        llvm::Value *Chunk) {

  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * local_size = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_size), arg);
  llvm::Value * local_size_casted = CGF.Builder.CreateTruncOrBitCast(local_size, CGF.Int32Ty);
  LValue LBLValue = CGF.MakeAddrLValue(LB, CGF.getContext().getIntPtrType());
  llvm::Value * lb = CGF.EmitLoadOfScalar(LBLValue, Loc);

  if(Chunk == nullptr) {
    /** Static Scheduling:
     * When no chunk_size is specified, the iteration space is divided into chunks
     * that are approximately equal in size, and at most one chunk is distributed to
     * each thread. The size of the chunks is unspecified in this case.
     */
    if(ScheduleKind.Schedule == OpenMPScheduleClauseKind::OMPC_SCHEDULE_static) {
      // here we do: chunksize = (ub-lb+local_size-1)/local_size
      LValue UBLValue = CGF.MakeAddrLValue(UB, CGF.getContext().getIntPtrType());
      llvm::Value *ub = CGF.EmitLoadOfScalar(UBLValue, Loc);
      llvm::Value *it_space = CGF.Builder.CreateSub(ub, lb);
      it_space = CGF.Builder.CreateAdd(it_space, local_size_casted);
      it_space = CGF.Builder.CreateSub(it_space, CGF.Builder.getInt32(1));
      Chunk = CGF.Builder.CreateUDiv(it_space, local_size_casted);
    } else {
      // If no schedule is specified, scheduling is implementation defined.
      // For the spir target, we choose schedule(static,1) as default
      Chunk = CGF.Builder.getInt32(1);
    }
  }

  llvm::CallInst * locid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value * ltid = CGF.Builder.CreateTruncOrBitCast(locid, CGF.Int32Ty);

  // lower bound is: provided lb + localthreadID (* chunksize)
  llvm::Value * lbdiff = CGF.Builder.CreateMul(ltid, Chunk);
  lb = CGF.Builder.CreateAdd(lb, lbdiff);
  CGF.EmitStoreOfScalar(lb, LBLValue, true);

  // upper bound is: lb + chunk-1 (for chunksize=1, this results in lb=ub)
  llvm::Value * ch = CGF.Builder.CreateSub(Chunk, CGF.Builder.getInt32(1));
  llvm::Value * ub = CGF.Builder.CreateAdd(lb, ch);
  CGF.EmitStoreOfScalar(ub, CGF.MakeAddrLValue(UB, CGF.getContext().getIntPtrType()), true);

  // stride is: local workgroup size (* chunksize)
  llvm::Value * stride = CGF.Builder.CreateMul(local_size_casted, Chunk);
  CGF.EmitStoreOfScalar(stride, CGF.MakeAddrLValue(ST, CGF.getContext().getIntPtrType()), true);
}

void CGOpenMPRuntimeSPIR::emitDistributeStaticInit(
        CodeGenFunction &CGF, SourceLocation Loc,
        OpenMPDistScheduleClauseKind SchedKind, unsigned IVSize, bool IVSigned,
        bool Ordered, Address IL, Address LB, Address UB, Address ST,
        llvm::Value *Chunk) {
  // take num_groups
  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * num_groups = CGF.EmitRuntimeCall(createRuntimeFunction(get_num_groups), arg);
  llvm::Value * num_groups_casted = CGF.Builder.CreateTruncOrBitCast(num_groups, CGF.Int32Ty);
  // take lb
  LValue LBLValue = CGF.MakeAddrLValue(LB, CGF.getContext().getIntPtrType());
  llvm::Value * lb = CGF.EmitLoadOfScalar(LBLValue, Loc);


  if(Chunk == nullptr) {
    // chunksize is unspecified: calculate a reasonable chunksize
    // chunksize should be multiple of local_size:
    llvm::CallInst *local_size = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_size), arg);
    llvm::Value *local_size_casted = CGF.Builder.CreateTruncOrBitCast(local_size, CGF.Int32Ty);

    if(SchedKind == OpenMPDistScheduleClauseKind::OMPC_DIST_SCHEDULE_static) {
      /* 2.10.8 distribute construct - dist_schedule(static)
       * When no chunk_size is specified, the iteration space is divided
       * into chunks that are approximately equal in size,
       * and at most one chunk is distributed to each team of the league.
       */
      // here we do: chunksize = ((((ub-lb+local_size-1)/local_size)+num_groups-1)/num_groups)*local_size
      LValue UBLValue = CGF.MakeAddrLValue(UB, CGF.getContext().getIntPtrType());
      llvm::Value *ub = CGF.EmitLoadOfScalar(UBLValue, Loc);
      llvm::Value *diff = CGF.Builder.CreateSub(ub, lb);
      llvm::Value *total = CGF.Builder.CreateSub(diff, CGF.Builder.getInt32(1));
      llvm::Value *num_blocks = CGF.Builder.CreateUDiv(total, local_size_casted);
      num_blocks = CGF.Builder.CreateAdd(num_blocks, num_groups_casted);
      num_blocks = CGF.Builder.CreateSub(num_blocks, CGF.Builder.getInt32(1));
      Chunk = CGF.Builder.CreateMul(CGF.Builder.CreateUDiv(num_blocks, num_groups_casted), local_size_casted);

    } else {
      // if not static, scheduling is implementation defined:
      // we just assign local_size as chunksize and do round-robin
      Chunk = local_size_casted;
    }
  }

  llvm::CallInst * gid = CGF.EmitRuntimeCall(createRuntimeFunction(get_group_id), arg);
  llvm::Value * gid_casted = CGF.Builder.CreateTruncOrBitCast(gid, CGF.Int32Ty);

  // lower bound is: lb + groupID * chunksize
  lb = CGF.Builder.CreateAdd(CGF.Builder.CreateMul(gid_casted, Chunk), lb);
  CGF.EmitStoreOfScalar(lb, LBLValue, true);

  // upper bound is: lb + chunksize-1
  llvm::Value * ub = CGF.Builder.CreateAdd(lb, CGF.Builder.CreateSub(Chunk, CGF.Builder.getInt32(1)));
  CGF.EmitStoreOfScalar(ub, CGF.MakeAddrLValue(UB, CGF.getContext().getIntPtrType()), true);

  // stride is: chunksize * num_groups
  llvm::Value * stride = CGF.Builder.CreateMul(Chunk, num_groups_casted);
  CGF.EmitStoreOfScalar(stride, CGF.MakeAddrLValue(ST, CGF.getContext().getIntPtrType()), true);
}

void CGOpenMPRuntimeSPIR::emitForStaticFinish(CodeGenFunction &CGF,
                                          SourceLocation Loc) {}

void CGOpenMPRuntimeSPIR::emitNumThreadsClause(CodeGenFunction &CGF,
                                           llvm::Value *NumThreads,
                                           SourceLocation Loc) {
  // only emit this clause if it is the outermost parallel construct
  if(inParallel)
    return;
  // principle: if(thread_id < NumThreads) {...}
  emitNumThreadsHeader(CGF, NumThreads);
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
  assert(false && "For SPIR target, dynamic dispatch is not supported.");
}

bool CGOpenMPRuntimeSPIR::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  // In case of OMPC_SCHEDULE_unknown we return false
  // as we want to emit schedule(static,1) if no schedule clause is specified
  // more precise: the case below is the only one, for which we partition the iteration space
  // into chunks of equal size only to be conformant with the specification
  return (ScheduleKind == OpenMPScheduleClauseKind ::OMPC_SCHEDULE_static && !Chunked);
}

bool CGOpenMPRuntimeSPIR::isStaticNonchunked(
        OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  return ScheduleKind == OpenMPDistScheduleClauseKind ::OMPC_DIST_SCHEDULE_static && !Chunked;
}

bool CGOpenMPRuntimeSPIR::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  // we don't support real dynamic scheduling and just emit everything as static
  return false;
}