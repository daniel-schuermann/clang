//===--- AMDGPU.h - AMDGPU ToolChain Implementations ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIR_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIR_H

#include "Gnu.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {

namespace spir {

  class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
  public:
      Linker(const ToolChain &TC);
      bool isLinkJob() const override;
      bool hasIntegratedCPP() const override;
      void ConstructJob(Compilation &C, const JobAction &JA,
                        const InputInfo &Output, const InputInfoList &Inputs,
                        const llvm::opt::ArgList &TCArgs,
                        const char *LinkingOutput) const override;
  };

} // end namespace spir
} // end namespace tools

namespace toolchains {

    class LLVM_LIBRARY_VISIBILITY SPIRToolChain final : public Generic_ELF {
    public:
        SPIRToolChain(const Driver &D, const llvm::Triple &Triple,
                    const llvm::opt::ArgList &Args);

        bool useIntegratedAs() const override { return true; }
        bool isCrossCompiling() const override { return true; }
        bool IsIntegratedAssemblerDefault() const override { return true; };
        bool HasNativeLLVMSupport() const override { return true; };

        bool IsMathErrnoDefault() const override { return false; };
        bool IsObjCNonFragileABIDefault() const override { return false; };
        bool UseObjCMixedDispatch() const override { return false; };
        bool isPICDefault() const override { return false; };
        bool isPIEDefault() const override { return false; };
        bool isPICDefaultForced() const override { return false; };

        bool hasBlocksRuntime() const override { return false; };
        bool SupportsObjCGC() const override { return false; };
        bool SupportsProfiling() const override { return false; };

    protected:
        Tool *buildLinker() const override;

    };

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIR_H
