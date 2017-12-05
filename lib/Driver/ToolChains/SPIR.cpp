//===--- AMDGPU.cpp - AMDGPU ToolChain Implementations ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SPIR.h"
#include "InputInfo.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "llvm/Option/ArgList.h"
#include <iostream>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

spir::Linker::Linker(const ToolChain &TC)
        : GnuTool("spir::Linker", "spirv-link", TC) {}

bool spir::Linker::isLinkJob() const {
  return true;
}

bool spir::Linker::hasIntegratedCPP() const {
  return false;
}

void spir::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  std::cout << "build Linker\n";

  std::string Linker = getToolChain().GetProgramPath(getShortName());
  ArgStringList CmdArgs;
  //CmdArgs.push_back("--target-env=opencl2.1");
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs, Inputs));
}


/// SPIR Toolchain

SPIRToolChain::SPIRToolChain(const Driver &D, const llvm::Triple &Triple,
                         const llvm::opt::ArgList &Args)
        : Generic_ELF(D, Triple, Args) {
  std::cout << "using SPIR ToolChain\n";
}


Tool *SPIRToolChain::buildLinker() const {
  return new tools::spir::Linker(*this);
}