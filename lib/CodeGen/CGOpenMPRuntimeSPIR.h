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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPIR_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPIR_H

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeSPIR : public CGOpenMPRuntime {

protected:
  enum OpenMPRTLFunctionSPIR {
    get_global_id,
    get_local_id,
    get_local_size,
    get_num_groups,
    get_group_id,
    work_group_barrier
  };
  // TODO: comment these functions!
  QualType getAddrSpaceType(QualType T, LangAS::ID AddrSpace);
  llvm::BasicBlock * MasterContBlock;
  void emitMasterHeader(CodeGenFunction &CGF);
  void emitMasterFooter(CodeGenFunction &CGF);
  void GenOpenCLArgMetadata(const RecordDecl *FD, llvm::Function *Fn, llvm::LLVMContext &Context, CGBuilderTy &Builder);
  llvm::Constant *createRuntimeFunction(OpenMPRTLFunctionSPIR Function);
  void setTargetParallel(OpenMPDirectiveKind kind);
  bool isTargetParallel;

public:
  explicit CGOpenMPRuntimeSPIR(CodeGenModule &CGM);

  /// \brief Emit outlined function for 'target' directive on the SPIR
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry,
                                  const RegionCodeGenTy &CodeGen) override;

  /// \brief Emits outlined function for the specified OpenMP parallel directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  virtual llvm::Value *emitParallelOutlinedFunction(
          const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
          OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen);

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  virtual void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                llvm::Value *OutlinedFn,
                                ArrayRef<llvm::Value *> CapturedVars,
                                const Expr *IfCond);

  /// \brief Emits outlined function for the specified OpenMP teams directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *emitTeamsOutlinedFunction(
          const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
          OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen);

  /// \brief Emits code for teams call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run by team masters. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitTeamsCall(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                     SourceLocation Loc, llvm::Value *OutlinedFn,
                     ArrayRef<llvm::Value *> CapturedVars) override;

  /// \brief Emits a master region.
  /// \param MasterOpGen Generator for the statement associated with the given
  /// master region.
  virtual void emitMasterRegion(CodeGenFunction &CGF,
                                const RegionCodeGenTy &MasterOpGen,
                                SourceLocation Loc);

  /// \brief Emit an implicit/explicit barrier for OpenMP threads.
  /// \param Kind Directive for which this implicit barrier call must be
  /// generated. Must be OMPD_barrier for explicit barrier generation.
  /// \param EmitChecks true if need to emit checks for cancellation barriers.
  /// \param ForceSimpleCall true simple barrier call must be emitted, false if
  /// runtime class decides which one to emit (simple or with cancellation
  /// checks).
  ///
  virtual void emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                               OpenMPDirectiveKind Kind,
                               bool EmitChecks = true,
                               bool ForceSimpleCall = false);

  /// \brief Call the appropriate runtime routine to initialize it before start
  /// of loop.
  ///
  /// This is used only in case of static schedule, when the user did not
  /// specify a ordered clause on the loop construct.
  /// Depending on the loop schedule, it is necessary to call some runtime
  /// routine before start of the OpenMP loop to get the loop upper / lower
  /// bounds \a LB and \a UB and stride \a ST.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param ScheduleKind Schedule kind, specified by the 'schedule' clause.
  /// \param IVSize Size of the iteration variable in bits.
  /// \param IVSigned Sign of the interation variable.
  /// \param Ordered true if loop is ordered, false otherwise.
  /// \param IL Address of the output variable in which the flag of the
  /// last iteration is returned.
  /// \param LB Address of the output variable in which the lower iteration
  /// number is returned.
  /// \param UB Address of the output variable in which the upper iteration
  /// number is returned.
  /// \param ST Address of the output variable in which the stride value is
  /// returned nesessary to generated the static_chunked scheduled loop.
  /// \param Chunk Value of the chunk for the static_chunked scheduled loop.
  /// For the default (nullptr) value, the chunk 1 will be used.
  ///
  virtual void emitForStaticInit(CodeGenFunction &CGF, SourceLocation Loc,
                                 const OpenMPScheduleTy &ScheduleKind,
                                 unsigned IVSize, bool IVSigned, bool Ordered,
                                 Address IL, Address LB, Address UB, Address ST,
                                 llvm::Value *Chunk = nullptr);

  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param SchedKind Schedule kind, specified by the 'dist_schedule' clause.
  /// \param IVSize Size of the iteration variable in bits.
  /// \param IVSigned Sign of the interation variable.
  /// \param Ordered true if loop is ordered, false otherwise.
  /// \param IL Address of the output variable in which the flag of the
  /// last iteration is returned.
  /// \param LB Address of the output variable in which the lower iteration
  /// number is returned.
  /// \param UB Address of the output variable in which the upper iteration
  /// number is returned.
  /// \param ST Address of the output variable in which the stride value is
  /// returned nesessary to generated the static_chunked scheduled loop.
  /// \param Chunk Value of the chunk for the static_chunked scheduled loop.
  /// For the default (nullptr) value, the chunk 1 will be used.
  ///
  virtual void emitDistributeStaticInit(CodeGenFunction &CGF, SourceLocation Loc,
                                        OpenMPDistScheduleClauseKind SchedKind,
                                        unsigned IVSize, bool IVSigned,
                                        bool Ordered, Address IL, Address LB,
                                        Address UB, Address ST,
                                        llvm::Value *Chunk = nullptr);

  /// \brief Call the appropriate runtime routine to notify that we finished
  /// all the work with current loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  ///
  virtual void emitForStaticFinish(CodeGenFunction &CGF, SourceLocation Loc);

  // unsupported
  virtual void emitForDispatchInit(CodeGenFunction &CGF, SourceLocation Loc,
                                   const OpenMPScheduleTy &ScheduleKind,
                                   unsigned IVSize, bool IVSigned, bool Ordered,
                                   const DispatchRTInput &DispatchValues);

  /// \brief Check if the specified \a ScheduleKind is static non-chunked.
  /// This kind of worksharing directive is emitted without outer loop.
  /// \param ScheduleKind Schedule kind specified in the 'schedule' clause.
  /// \param Chunked True if chunk is specified in the clause.
  ///
  virtual bool isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                  bool Chunked) const;

  /// \brief Check if the specified \a ScheduleKind is static non-chunked.
  /// This kind of distribute directive is emitted without outer loop.
  /// \param ScheduleKind Schedule kind specified in the 'dist_schedule' clause.
  /// \param Chunked True if chunk is specified in the clause.
  ///
  virtual bool isStaticNonchunked(OpenMPDistScheduleClauseKind ScheduleKind,
                                  bool Chunked) const;

  /// \brief Check if the specified \a ScheduleKind is dynamic.
  /// This kind of worksharing directive is emitted without outer loop.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  ///
  virtual bool isDynamic(OpenMPScheduleClauseKind ScheduleKind) const;

};

} // CodeGen namespace.
} // clang namespace.

#endif //LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPIR_H
