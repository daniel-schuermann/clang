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
//#include "CGOpenMPRuntime.cpp"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclOpenMP.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include <iostream>

using namespace clang;
using namespace CodeGen;

CGOpenMPRuntimeSPIR::CGOpenMPRuntimeSPIR(CodeGenModule &CGM)
        : CGOpenMPRuntime(CGM) {
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

void enterAction() {
  std::cout << "enter\n";
}

void exitAction() {
  std::cout << "exit\n";
}

//FD->setType(CGM.getContext().getPointerType(CGM.getContext().getAddrSpaceQualType(
//        ArgType.getTypePtr()->getPointeeType(), LangAS::opencl_global)));
QualType CGOpenMPRuntimeSPIR::getAddrSpaceType(QualType T, LangAS::ID AddrSpace) {
  T.getTypePtr()->dump();
  if(T.getTypePtr()->isAnyPointerType() || T.getTypePtr()->isLValueReferenceType()) {
    return CGM.getContext().getPointerType(getAddrSpaceType(T.getTypePtr()->getPointeeType(), AddrSpace));
  } else {
    return CGM.getContext().getAddrSpaceQualType(T, AddrSpace);
  }
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

  CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());


  //CS.getCapturedDecl()->dump();
  //const RecordDecl *RD = CS.getCapturedRecordDecl();
//----------------
  /*
  D.dump();
  for (auto param : CS.getCapturedDecl()->parameters()) {
    param->dump();
  }
  for (auto &I : CS.captures()) {
    VarDecl * var = I.getCapturedVar();
    QualType ArgType = var->getType();
    //var->getType().dump();
    if ((I.capturesVariableByCopy() && !ArgType->isAnyPointerType()) ||
        I.capturesVariableArrayType()) {
      //ArgType.dump();
      //ArgType = CGM.getContext().getUIntPtrType();
      ArgType.dump();
    }

    //ArgType.dump();
    if(const Type * t = ArgType.getTypePtrOrNull()) {
      t->getPointeeOrArrayElementType()->dump();
    }

    ArgType.dump();

    var->setType(CGM.getContext().getPointerType(CGM.getContext().getAddrSpaceQualType(
            ArgType,  LangAS::opencl_global)));
    var->dump();
  }
   */
  //-----------------
  const RecordDecl *RD = CS.getCapturedRecordDecl();
  auto I = CS.captures().begin();
    for (auto *FD : RD->fields()) {

      // TODO: are we sure to always have a Pointer here?
      QualType ArgType = FD->getType();
      /*
      if ((I->capturesVariableByCopy() && !ArgType->isAnyPointerType()) ||
          I->capturesVariableArrayType()) {
        //ArgType.dump();
        ArgType = CGM.getContext().getUIntPtrType();
        ArgType.dump();
      } else {
        FD->setType(CGM.getContext().getPointerType(CGM.getContext().getAddrSpaceQualType(
                ArgType.getTypePtr()->getPointeeType(), LangAS::opencl_global)));
      }
       */
      FD->setType(getAddrSpaceType(ArgType, LangAS::opencl_global));
      //FD->dump();
      ++I;
    }

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);

  OutlinedFn->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  OutlinedFn->addFnAttr(llvm::Attribute::NoUnwind);
  OutlinedFn->getFunctionType()->dump();


  CodeGenFunction CGF(CGM);
  GenOpenCLArgMetadata(CS.getCapturedRecordDecl(), OutlinedFn,
                          OutlinedFn->getContext(), CGF.Builder);


  // remove optnone

}

llvm::Value *CGOpenMPRuntimeSPIR::emitParallelOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_parallel);

  // TODO: should we directly emit the body?
  //CodeGenFunction CGF(CGM, true);
  //llvm::BasicBlock * BB = CGF.createBasicBlock(".parallel");

  const RecordDecl *RD = CS->getCapturedRecordDecl();
  for (auto *FD : RD->fields()) {
    FD->setType(getAddrSpaceType(FD->getType(), LangAS::opencl_global));
  }

  return CGOpenMPRuntime::emitParallelOutlinedFunction(D, ThreadIDVar, InnermostKind, CodeGen);
}

void CGOpenMPRuntimeSPIR::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars,
                                       const Expr *IfCond) {

  if (!CGF.HaveInsertPoint())
    return;

  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }

  // TODO: if we are in teams region, we can reuse global and local id!
  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * gtid = CGF.EmitRuntimeCall(createRuntimeFunction(get_global_id), arg);
  Address global_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".gtid");
  llvm::Value * gtid_casted = CGF.Builder.CreateTruncOrBitCast(gtid, CGF.Int32Ty);
  CGF.EmitStoreOfScalar(gtid_casted, CGF.MakeAddrLValue(global_tid, CGF.getContext().getIntPtrType()), true);

  llvm::CallInst * ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  Address local_tid = CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4), ".btid");
  llvm::Value * ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  CGF.EmitStoreOfScalar(ltid_casted, CGF.MakeAddrLValue(local_tid, CGF.getContext().getIntPtrType()), true);

  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(global_tid.getPointer()); // global_tid
  OutlinedFnArgs.push_back(local_tid.getPointer());  // bound_tid
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}


llvm::Value *CGOpenMPRuntimeSPIR::emitTeamsOutlinedFunction(
        const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
        OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {


  const CapturedStmt *CS = D.getCapturedStmt(OMPD_teams);


  const RecordDecl *RD = CS->getCapturedRecordDecl();
  auto I = CS->captures().begin();
  for (auto *FD : RD->fields()) {
    // TODO: are we sure to always have a Pointer here?
    QualType ArgType = FD->getType();
    if ((I->capturesVariableByCopy() && !ArgType->isAnyPointerType()) ||
        I->capturesVariableArrayType()) {
      //ArgType.dump();
      ArgType = CGM.getContext().getUIntPtrType();
      ArgType.dump();
    } else {
      FD->setType(CGM.getContext().getPointerType(CGM.getContext().getAddrSpaceQualType(
              ArgType.getTypePtr()->getPointeeType(), LangAS::opencl_global)));
    }
    FD->dump();
    ++I;
  }

  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  //CodeGenFunction CGF(CGM, true);
  //bool HasCancel = false;
  //CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
  //                                  HasCancel, getOutlinedHelperName());
  //CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  //return CGF.GenerateOpenMPCapturedStmtFunction(*CS);

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

  //llvm::Constant * getGlobalId = CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.SizeTy, CGM.Int32Ty), "get_global_id");
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
  OutlinedFnArgs.push_back(global_tid.getPointer()); // TODO: global_tid
  OutlinedFnArgs.push_back(local_tid.getPointer()); // TODO: bound_tid
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}

void CGOpenMPRuntimeSPIR::emitMasterRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &MasterOpGen,
                                       SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  // principle: if(threadID == 0):
  llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
  llvm::CallInst * ltid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
  llvm::Value * ltid_casted = CGF.Builder.CreateTruncOrBitCast(ltid, CGF.Int32Ty);
  llvm::Value * cond = CGF.Builder.CreateIsNull(ltid_casted);
  llvm::BasicBlock * ThenBlock = CGF.createBasicBlock("omp_if.then");
  llvm::BasicBlock * ContBlock = CGF.createBasicBlock("omp_if.end");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(cond, ThenBlock, ContBlock);
  CGF.EmitBlock(ThenBlock);

  emitInlinedDirective(CGF, OMPD_master, MasterOpGen);
  CGF.EmitBranch(ContBlock);
  CGF.EmitBlock(ContBlock, true);
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

    assert(ScheduleKind.Schedule == OpenMPScheduleClauseKind::OMPC_SCHEDULE_static);
    llvm::Value * arg[] = { CGF.Builder.getInt32(0) };
    llvm::CallInst * local_size = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_size), arg);
    llvm::Value * local_size_casted = CGF.Builder.CreateTruncOrBitCast(local_size, CGF.Int32Ty);
    LValue LBLValue = CGF.MakeAddrLValue(LB, CGF.getContext().getIntPtrType());
    llvm::Value * lb = CGF.EmitLoadOfScalar(LBLValue, Loc);

    /**
     * When no chunk_size is specified, the iteration space is divided into chunks
     * that are approximately equal in size, and at most one chunk is distributed to
     * each thread. The size of the chunks is unspecified in this case.
     */
    if(Chunk == nullptr) {
      // here we do: chunksize = (ub-lb+local_size-1)/local_size
      LValue UBLValue = CGF.MakeAddrLValue(UB, CGF.getContext().getIntPtrType());
      llvm::Value * ub = CGF.EmitLoadOfScalar(UBLValue, Loc);
      llvm::Value * it_space = CGF.Builder.CreateSub(ub, lb);
      it_space = CGF.Builder.CreateAdd(it_space, local_size_casted);
      it_space = CGF.Builder.CreateSub(it_space, CGF.Builder.getInt32(1));
      Chunk = CGF.Builder.CreateUDiv(it_space, local_size_casted);
    }

    llvm::CallInst * locid = CGF.EmitRuntimeCall(createRuntimeFunction(get_local_id), arg);
    llvm::Value * ltid = CGF.Builder.CreateTruncOrBitCast(locid, CGF.Int32Ty);

    // lower bound is: provided lb + localthreadID (* chunksize)
    llvm::Value * lbdiff = CGF.Builder.CreateMul(ltid, Chunk);
    lb = CGF.Builder.CreateAdd(lb, lbdiff);
    CGF.EmitStoreOfScalar(lb, LBLValue, true);

    // upper bound is: lb + chunk-1 (for chunksize=1, this results in lb=ub)
    lb = CGF.Builder.CreateSub(lb, CGF.Builder.getInt32(1));
    llvm::Value * ub = CGF.Builder.CreateAdd(lb, Chunk);
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