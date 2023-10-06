#include <fstream>
#include <map>
#include <thread>

#include "mutator_lib.h"

#include <llvm/Pass.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/CommandLine.h>
#include <iostream>

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"



using namespace llvm;

#define DEBUG_TYPE "mutator"

static cl::opt<std::string> Mutation("mutation_pattern",
                                   cl::desc("the source location and mutation pattern"),
                                   cl::value_desc("string"));
static cl::opt<bool> CPP("cpp", cl::desc("Enable CPP-only mutations"));

namespace {
//counter for method calls, each method call gets a unique ID
    int callIDCounter = 1;

// a counter and the number of functions to print the current status
    int number_functions = 0;
    int funcounter = 0;

// the following variables define the location of the mutation as well as the pattern
// containing in order: Directory, File, line, column, mutation-ID as strings
    using json = nlohmann::json;
    json seglist;

    class Worker {
    private:

        std::mutex &builderMutex;
        llvm::Module &M;

    public:

        explicit Worker() = delete;

        Worker(Module &M, std::mutex &builderMutex, std::mutex &fileMutex)
                : builderMutex(builderMutex), M(M) {

        }

        /**
         * Instrument all functions given as parameter.
         * @param functions
         */
        void instrumentFunctions(std::vector<Function *> functions) {
            for (auto F: functions) {
//            builderMutex.lock();
//            std::cout << "[INFO] in thread " << std::this_thread::get_id() << ": "
//                      << "instrumenting function " << ++funcounter << " of " << number_functions
//                      << ": " << F->getName().data()
//                      << std::endl;
//            builderMutex.unlock();

                findPatternInFunction(*F);
            }
        }


        /**
         * Instrument the given instruction with the given builders.
         * @param instr
         * @param builder
         * @param nextInstructionBuilder
         */
        void handInstructionToPatternMatchers(Instruction *instr, int id, IRBuilder<> *builder,
                                              IRBuilder<> *nextInstructionBuilder) {
            // Call instructions are handled differently
            if (auto *callinst = dyn_cast<CallInst>(instr)) {
                Function *fun = callinst->getCalledFunction();
                if (fun != nullptr && fun->isIntrinsic() && !dyn_cast<MemCpyInst>(callinst) &&
                    !dyn_cast<MemMoveInst>(callinst)
                    && !dyn_cast<VAStartInst>(callinst) && !dyn_cast<VAArgInst>(callinst) &&
                    !dyn_cast<VACopyInst>(callinst)
                    && !dyn_cast<VAEndInst>(callinst)) {
                    // skip llvm intrinsic functions other than llvm.memcpy and llvm memmove
                    return;
                }
            }
//            std::string instructionString;
//            llvm::raw_string_ostream os(instructionString);
//            instr->print(os);
            // NEVER delete any instruction other than the one we are currently working on, this may result in undef behavior!
            // If at all do a look-back and change things in already touched instructions.
            if (mutatePattern(builder, nextInstructionBuilder, instr, id, builderMutex, M)) {
//                std::string instructionString;
//                llvm::raw_string_ostream os(instructionString);
//                instr->print(os);
//                std::cout << "[INFO C] Applied mutation on instruction " << os.str() << "\n";
            }
        }


        /**
         * Instrument one function, i.e. run over all instructions in that function and instrument them.
         * @param F the given function
         * @return true on successful instrumentation
         */
        bool findPatternInFunction(Function &F) {
            std::vector<Instruction *> toInstrument;
            std::vector<int> fID;
            int counter = 0;
            for (BasicBlock &bb : F) {
                auto first_insertion_point = bb.getFirstInsertionPt();

                for (BasicBlock::iterator itr_bb = first_insertion_point; itr_bb != bb.end(); ++itr_bb) {
                    toInstrument.push_back(&*itr_bb);
                    fID.push_back(counter++);
                }
            }
            counter = 0;
            for (Instruction *instr : toInstrument) {
                BasicBlock::iterator itr_bb(instr);
                builderMutex.lock();
                IRBuilder<> builder(instr->getParent(), itr_bb);
                IRBuilder<> nextInstructionBuilder(instr->getParent(), std::next(itr_bb, 1));
                builderMutex.unlock();
                handInstructionToPatternMatchers(instr, fID[counter++], &builder, &nextInstructionBuilder);
            }

            return true;
        }
    };

    struct MutatorPlugin : public llvm::PassInfoMixin<MutatorPlugin> {
        // Without isRequired returning true, this pass will be skipped for functions
        // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
        // all functions with optnone.
        static bool isRequired() { return true; }

        PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {

            std::cout << "MutationFinder Pass" << std::endl;

            bool changed =  runOnModule(M);

            return (changed ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all());
        }

        bool runOnModule(Module &M) {
            auto &llvm_context = M.getContext();

            // TODO read mutation patterns



            std::mutex builderMutex;
            std::mutex fileMutex;
            // std::cout << "[INFO C] Mutating: " << Mutation << "\n";
            seglist = json::parse(Mutation);
            populatePatternVectors(&seglist);
            insertMutationApiFunctions(M, CPP);
            //Parsing the string into a json
            std::string segment;
            unsigned int concurrentThreadsSupported = 1; //ceil(std::thread::hardware_concurrency());
//        std::cout << "[INFO] number of threads: " << concurrentThreadsSupported << std::endl;

            std::vector<std::vector<Function *>> threadFunctions(concurrentThreadsSupported);
            auto i = 0;
            for (Function &f : M.functions()) {
                if (f.isDeclaration()) {
                    continue;
                }

                threadFunctions[i % concurrentThreadsSupported].push_back(&f);
                ++i;
            }

            number_functions = i;
            std::vector<std::thread> threads;
            for (auto &functions : threadFunctions) {
                threads.push_back(
                        std::thread(&Worker::instrumentFunctions, new Worker(M, builderMutex, fileMutex), functions));
            }

            for (auto &thread : threads) {
                thread.join();
            }
            return true;
        }
    };
}

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MutationPlugin", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [&](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if(Name == "mutatorplugin") {
                        MPM.addPass(MutatorPlugin());
                        return true;
                    }
                    return false;
                });
          }};
}
