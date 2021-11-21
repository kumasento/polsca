# A script to run the c baseline of polybench
# Usage: python3 polybench-base.py [options]
# Options:
#   -h, --help            show this help message and exit
#   -p SIZE, --problem-size=SIZE
#                         Problem size, default SMALL
#   -j JOBS, --jobs=JOBS  jobs, default = 1
#   -d WORKDIR, --work-dir=WORKDIR
#                         working directory, default = ./tmp
#   -e EXAMPLE, --example=EXAMPLE
#                         example, default = 2mm
#   -s VITIS, --vitis=VITIS
#                         vitis_hls path, default = vitis_hls
#   -t, --run-vitis       Run Vitis HLS
#   -c, --cosim           Cosimulation
#   -r, --reset           Reset. Clean all the files

from multiprocessing import Process, Queue
from optparse import OptionParser
import sys, os, time, datetime, multiprocessing
# from typing import List

#############################################################
#    Polybench Setup
#############################################################

PolybenchSizes = ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"]
PolybenchList = [
"datamining/correlation",
"datamining/covariance",
"linear-algebra/kernels/2mm",
"linear-algebra/kernels/3mm",
"linear-algebra/kernels/atax",
"linear-algebra/kernels/bicg",
"linear-algebra/kernels/doitgen",
"linear-algebra/kernels/mvt",
"linear-algebra/blas/gemm",
"linear-algebra/blas/gemver",
"linear-algebra/blas/gesummv",
"linear-algebra/blas/symm",
"linear-algebra/blas/syr2k",
"linear-algebra/blas/syrk",
"linear-algebra/blas/trmm",
"linear-algebra/solvers/cholesky",
"linear-algebra/solvers/durbin",
"linear-algebra/solvers/gramschmidt",
"linear-algebra/solvers/lu",
"linear-algebra/solvers/ludcmp",
"linear-algebra/solvers/trisolv",
"medley/deriche",
"medley/floyd-warshall",
"medley/nussinov",
"stencils/adi",
"stencils/fdtd-2d",
"stencils/heat-3d",
"stencils/jacobi-1d",
"stencils/jacobi-2d",
"stencils/seidel-2d"
]

def getBenchmark(example):
    for polybench in PolybenchList:
        if polybench[polybench.rfind("/")+1:] == example:
            return PolybenchList.index(polybench)
    return -1

#############################################################
#    TCL Gen
#############################################################

def TCLGenOnce(index, workDir, size, cosim):
    benchmark = os.path.basename(PolybenchList[index])
    benchmarkDir = os.path.join(workDir, PolybenchList[index])
    src = os.path.join(benchmarkDir, benchmark)
    baseTcl = """
open_project -reset cbase
add_files {src}.c -cflags "-I {srcDir} -I {workDir}/utilities -D{size}_DATASET" -csimflags "-I {srcDir} -I {workDir}/utilities -D{size}_DATASET"
add_files -tb {{ {src}.c {workDir}/utilities/polybench.c }} -cflags "-I {srcDir} -I {workDir}/utilities -D{size}_DATASET" -csimflags "-I {srcDir} -I {workDir}/utilities -D{size}_DATASET" 
set_top kernel_{benchmark}
open_solution -reset solution1
set_part "xqzu29dr-ffrf1760-1-i"
create_clock -period "100MHz"
config_compile -pipeline_loops 16
csynth_design
{cosim}
exit
""".format(
src = src,
srcDir = benchmarkDir,
workDir = workDir,
size = size,
benchmark = benchmark.replace("-", "_"),
cosim = cosim
)
    with open(os.path.join(benchmarkDir, "base.tcl"), 'w') as f:
        f.write(baseTcl)
    os.system('sed -i \'s/static//g\' "{}"'.format(src+".c"))

def TCLGenJob(jobID, jobs, workDir, size, cosim, polybenchCount):
    jobs = int(multiprocessing.cpu_count()/jobs)
    for benchID in range(0, polybenchCount):
        if benchID % (jobID + 1) != jobID:
            continue
        TCLGenOnce(benchID, workDir, size, cosim)
        
def TCLGen(jobs, workDir, size, cosim):
    if jobs > multiprocessing.cpu_count():
        jobs = multiprocessing.cpu_count()

    polybenchCount = len(PolybenchList)
    starts = [None] * jobs
    ends = [None] * jobs
    unit = polybenchCount/jobs
    for i in range(0, jobs):
        starts[i] = int(i*unit)
        ends[i] = min(int((i+1)*unit), polybenchCount)
    
    threads = [None] * jobs
    queue = Queue()
    for i in range(0, jobs):
        threads[i] = Process(target=TCLGenJob, args=(i, jobs, workDir, size, cosim, polybenchCount))
        threads[i].start()
    for i in range(0, jobs):
        threads[i].join()

#############################################################
#    Run Vitis
#############################################################

def RunVitisOnce(index, workDir, printLog, vitis):
    benchmark = os.path.basename(PolybenchList[index])
    benchmarkDir = os.path.join(workDir, PolybenchList[index])
    if printLog:
        os.system('(cd {}; {} {})'.format(
            benchmarkDir, vitis, os.path.join(workDir, PolybenchList[index], "base.tcl")))
    else:
        os.system('(cd {}; {} {} > /dev/null; echo "Finished benchmark: {}")'.format(
            benchmarkDir, vitis, os.path.join(workDir, PolybenchList[index], "base.tcl"), benchmark))

def RunVitisJob(jobID, jobs, workDir, polybenchCount, vitis):
    jobs = int(multiprocessing.cpu_count()/jobs)
    for benchID in range(0, polybenchCount):
        if benchID % (jobID + 1) != jobID:
            continue
        RunVitisOnce(benchID, workDir, False, vitis)
        
def RunVitis(jobs, workDir, vitis):
    if jobs > multiprocessing.cpu_count():
        jobs = multiprocessing.cpu_count()

    polybenchCount = len(PolybenchList)
    starts = [None] * jobs
    ends = [None] * jobs
    unit = polybenchCount/jobs
    for i in range(0, jobs):
        starts[i] = int(i*unit)
        ends[i] = min(int((i+1)*unit), polybenchCount)
    
    threads = [None] * jobs
    queue = Queue()
    for i in range(0, jobs):
        threads[i] = Process(target=RunVitisJob, args=(i, jobs, workDir, polybenchCount, vitis))
        threads[i].start()
    for i in range(0, jobs):
        threads[i].join()

    errCount = 0
    for polybench in PolybenchList:
        benchmarkLog = os.path.join(workDir, polybench, "vitis_hls.log")
        if 'ERROR' in open(benchmarkLog).read():
            print("Benchmark {} FAILED".format(os.path.basename(polybench)))
            errCount = errCount + 1
    print("Error: {}/{}".format(errCount, polybenchCount))

#############################################################
#    Options
#############################################################

def confirm():
    yes = {'yes','y', 'ye'}
    no = {'no','n'}

    choice = raw_input("This run will clean all the previously generated files. Are you sure you want to do this? [y/n] ").lower() if sys.version_info[0] < 3 else input("This run will clean all the previously generated files. Are you sure you want to do this? [y/n] ").lower()
    while choice not in yes and choice not in no:
        choice = raw_input("Please respond with 'y' or 'n'").lower() if sys.version_info[0] < 3 else input("Please respond with 'y' or 'n'").lower()

    if choice in yes:
        return True
    else:
        return False

def main():
    
    optparser = OptionParser()
    optparser.add_option("-p", "--problem-size", dest="size",
                         default="SMALL", help="Problem size, default SMALL")
    optparser.add_option("-j", "--jobs", dest="jobs",
                         default=1, help="jobs, default = 1")
    optparser.add_option("-d", "--work-dir", dest="workDir",
                         default="tmp", help="working directory, default = ./tmp")
    optparser.add_option("-e", "--example", dest="example",
                         help="example, default = 2mm")
    optparser.add_option("-s", "--vitis", dest="vitis",
                         default="vitis_hls",help="vitis_hls path, default = vitis_hls")
    
    optparser.add_option("-t", "--run-vitis", action="store_true", dest="run_vitis", default=True,
                         help="Run Vitis HLS")
    optparser.add_option("-c", "--cosim", action="store_true", dest="cosim", default=False,
                         help="Cosimulation")
    # optparser.add_option("-z", "--report", action="store_true", dest="evaluate", default=False,
    #                      help="Report the results")
    optparser.add_option("-r", "--reset", action="store_true", dest="reset", default=True,
                         help="Reset. Clean all the files")
    
    (options, args) = optparser.parse_args()

    workDir = os.path.join(os.getcwd(), options.workDir)
    phismDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    print("Benchmark directory is: {}".format(workDir))
    print("Phism directory is: {}".format(phismDir))
    print("Running in {} jobs".format(options.jobs))
    jobs = max(int(options.jobs), 1)
    cosim = "cosim_design" if options.cosim else "# cosim_design"
    size = options.size
    if size not in PolybenchSizes:
        raise IOError("Invalid size: {}".format(size))
    example = options.example
    vitis = options.vitis

    start = time.time()
    if os.path.exists(workDir) and options.reset:
        if confirm():
            os.system("rm -rf " + workDir)
        else:
            sys.exit()

    if not os.path.exists(workDir):
        os.system("mkdir -p " + workDir)
        os.system("cp -r "+os.path.join(phismDir, "example", "polybench", "*") + " " + workDir)

    if example:
        index = getBenchmark(example)
        if index == -1:
            raise IOError("Invalid example: {}".format(example))
        print("Running test for benchmark: {}".format(example))
        TCLGenOnce(index, workDir, size, cosim)
    else:
        TCLGen(jobs, workDir, size, cosim)

    if options.run_vitis:
        if example:
            RunVitisOnce(index, workDir, True, vitis)
        else:
            RunVitis(jobs, workDir, vitis)

    end = time.time()
    print("Total Time: "+"{:.2f}".format(end-start)+"s\n")

if __name__ == '__main__':
    main()
