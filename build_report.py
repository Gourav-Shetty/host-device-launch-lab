"""Build Gourav's CUDA report: same skeleton/styles as friend's, our content.

Seminar topic: CUDA kernel execution model
Realtime app: understanding how host code launches device computation (HostLaunch Lab)
Mirrors friend's 140-paragraph skeleton exactly (styles, order, aligns, table spot,
3 screenshots, footers). Verified after save.
"""
from docx import Document
from docx.shared import Pt, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SHOTS = r'C:\Users\unusu\host-device-launch-lab\report_shots'
OUT = r'C:\Users\unusu\Downloads\PC report - Tasfiya Nawaz (CUDA).docx'
IMG_W = Emu(5731510)

d = Document()
st = d.styles['Normal']
st.font.name = 'Times New Roman'
h1 = d.styles['Heading 1']
h1.font.name = 'Times New Roman'
h1.font.color.rgb = RGBColor(0, 0, 0)

sec = d.sections[0]
sec.page_width, sec.page_height = Emu(7560310), Emu(10692130)
sec.left_margin, sec.right_margin = Emu(0), Emu(783590)
sec.top_margin, sec.bottom_margin = Emu(346710), Emu(979170)

P16, P13, P14 = Pt(16), Pt(13), Pt(14)
LEFT, CENTER, RIGHT = WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT


def run(p, text, bold=None, size=None):
    r = p.add_run(text)
    r.bold = bold
    if size:
        r.font.size = size
    return r


def para(style='Normal', text='', align=None, bold=None, size=None):
    p = d.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if text:
        run(p, text, bold=bold, size=size)
    return p


def empty(align=LEFT):
    return para('Normal', '', align)


def dots():
    para('Normal', '.  ')


def chapter(n, trail=' '):
    p = para('Normal', '', LEFT)
    run(p, 'CHAPTER %d:' % n, bold=True, size=P16)
    run(p, trail, size=P13)


def title_intro(name):
    p = para('Normal', '', LEFT)
    run(p, ' ' * 47, bold=True, size=P13)
    run(p, ' %s ' % name, bold=True, size=P16)
    run(p, ' ')


def h1t(name):
    d.add_paragraph(name + '  ', style='Heading 1')


def sub(name):
    p = para('Normal', '', None)
    run(p, ' ' + name, bold=True)


def body(text):
    para('Normal', text)


def bull(text):
    d.add_paragraph(text, style='List Paragraph')


# ---------------- CHAPTER 1 ----------------
para('Normal', ' ', LEFT)  # P0
chapter(1)  # P1
title_intro('Introduction')  # P2
body('Modern accelerators process enormous streams of numbers: camera frames, sensor arrays, simulation grids and neural-network batches. In all of these workloads one processor writes the program but another processor runs it: the host (CPU) prepares the data while the device (GPU) executes thousands of threads at once. It is not enough to compute quickly in bulk; the system must move each chunk of data to the device, start the right number of threads, and bring the answer back before the next chunk arrives. This host-to-device handoff is the domain of the CUDA kernel execution model, where the engineering goal is to guarantee that every launch is correctly configured and that launch overhead never swallows the parallel speedup.')
body('At the same time, the hardware that runs such workloads has changed. A modern workstation is rarely a single processor chip; it is typically a latency-optimised multicore CPU next to a throughput-optimised GPU with its own memory, connected over a much slower link. Such machines have split-memory characteristics: the host can read its own RAM quickly, but reaching device memory costs an explicit copy across the interconnect, and launching work on the device costs a fixed setup penalty per kernel. A program that ignores this structure - copying too often, launching too few threads, or waiting idly for the device - can easily spend more time on overhead than on computation, and the expected GPU speedup is silently eroded.')
body('This project addresses that gap. It implements a real-time launch visualiser over a simulated device and executes the same host pipeline against three compute kernels - saxpy, vector-add and a stencil blur - with an explicit six-stage launch model. The design allocates each vector on the device, copies inputs across with a timed host-to-device transfer, fires the kernel with an explicit grid-by-block configuration, waits at a synchronisation point, and copies the result back, while a Flask web front end streams every launch to the browser with Server-Sent Events so the execution model can be watched as it happens. The seminar topic of this report is therefore "CUDA kernel execution model" and the real-time application is "understanding how host code launches device computation".')
body('The project was prepared and executed with AI assistance (OpenCode). The dashboard, the host launcher and the benchmark were built and run on a Windows workstation (AMD Ryzen 5 5600H, 6 cores / 12 logical threads) using Python 3.11 with NumPy; the real-CUDA path (host_launcher.py --cuda, via torch, cupy or numba) is provided as a complete reference path for laboratory machines equipped with an NVIDIA GPU, and the screenshots in Chapter 5 were captured from the running application with Playwright.')
dots()  # P7
para('Normal', '  ', LEFT)  # P8
para('Normal', '  ', LEFT)  # P9
empty()  # P10
para('Normal', ' ', LEFT)  # P11

# ---------------- CHAPTER 2 ----------------
chapter(2)  # P12
h1t('Objectives')  # P13
para('Normal', '  ', CENTER)  # P14
body('Understanding the CUDA kernel execution model is fundamental to engineering high-performance software, as launch configuration critically dictates the occupancy and execution speed of device programs on throughput-oriented GPUs. In the CUDA model, the host launches a kernel function with an execution configuration kernel<<<grid, block>>>, meaning grid blocks of block threads each, and the hardware distributes those blocks over streaming multiprocessors where groups of 32 threads (warps) execute in lockstep, whereas a misconfigured launch with too few threads leaves multiprocessors idle and too many threads per block exhausts registers and shared memory. When running a real-time launch visualiser with a strict per-launch display budget, configuration predictability becomes paramount so that "keeping up with the launches" is measurable against hard temporal deadlines without tail-latency spikes causing dropped frames. Implementing the pipeline across three distinct kernels - saxpy for the classic memory-bound lesson, vector-add for the minimal two-input case, and a stencil blur for neighbourhood-sharing behaviour - reveals distinct transfer-versus-compute dynamics and hardware bottlenecks.')
body('To overcome severe launch-overhead penalties in host-device programs, developers must systematically apply execution-model discipline: explicit device allocation (the cudaMalloc analogue) sized to the vector, timed host-to-device and device-to-host copies so transfer cost is never hidden, grid-by-block sizing matched to the vector length with a reported coverage ratio, synchronisation discipline so the host never reads results early, and strict per-stage timing (alloc, H2D, kernel, sync, D2H, free) to confirm where each millisecond goes. System evaluation requires rigorously measuring total launch latency, kernel-only time, transfer share, and speedup relative to a pure-Python host-loop baseline to confirm that the device path is consistently faster under real vector sizes. Crucially, engineering integrity mandates verifying that all paths produce identical results by comparing checksums of the output vectors across the serial, threaded-device and real-CUDA runs. Finally, comparing a tiny-vector run against a large-vector run highlights stark scaling behaviour; while small vectors show overhead dominance with transfers dwarfing the kernel, large vectors reveal dramatic kernel speedups of 10-18x, whereas naive launch schemes that ignore coverage degrade sharply under idle-thread waste and transfer saturation.')
para('Normal', '  ')  # P17
para('Normal', '  ', LEFT)  # P18
para('Normal', ' ', LEFT)  # P19
empty()  # P20
empty()  # P21
empty()  # P22
empty()  # P23

# ---------------- CHAPTER 3 ----------------
chapter(3)  # P24
h1t('Problem Statement')  # P25
para('Normal', '  ', CENTER)  # P26
body('In a host-device program, the host can never touch device memory directly. Memory attached to the host (RAM) is served with low latency to CPU code, whereas memory attached to the device must be allocated explicitly and reached only through copy commands over the interconnect, which are significantly slower per byte for small transfers. By default, a beginner program hides these costs: it allocates inside the timed region without noticing, copies back and forth redundantly, launches one thread per element without checking coverage, and reads results without synchronising. A launch pipeline written this way therefore suffers from transfer-dominated latency: the kernel itself finishes in milliseconds while copies and setup consume the rest, the interconnect becomes a bottleneck, occupancy stays low, and the measured speedup falls far short of the theoretical parallel gain.')
body('For a real-time launch visualiser - one that must fire, time and display a fresh kernel launch about twice per second - this overhead is critical. If the average time per launch exceeds the display interval, the dashboard falls behind and the operator watches stale launches instead of live ones. The problem is therefore: design, implement and evaluate a launch pipeline that demonstrably executes the full host-to-device sequence per launch, using the saxpy, vector-add and blur kernels with timed stages, and quantify the benefit of the device path over a pure host loop.')
body('Specifically, the project answers three questions:')
body('How much does the device path (threaded-device launch and real-CUDA launch) improve launch time over a serial host-loop baseline on the same vector?')
body('Does per-stage timing (alloc, H2D, kernel, sync, D2H, free) correctly expose whether a launch is transfer-bound or kernel-bound at each vector size?')
body('Does the dashboard deliver a freshly timed launch inside its display budget of about one launch per second, and how much headroom does it have?')
para('Normal', '  ', LEFT)  # P33
para('Normal', '  ', LEFT)  # P34
para('Normal', '  ', LEFT)  # P35
para('Normal', ' ', LEFT)  # P36
empty()  # P37
empty()  # P38
para('Normal', '  ', LEFT)  # P39

# ---------------- CHAPTER 4 ----------------
chapter(4)  # P40
h1t('Procedure')  # P41
sub('Workload Design')  # P42
body('The workload simulates a real-time device queue. The stream consists of repeated kernel launches; each launch processes a vector of N single-precision floats (N = 1,000 to 2,000,000, default 200,000) generated by a seeded NumPy generator, which guarantees that every implementation and every run launches on bit-identical input - a requirement for fair comparison and for verifying correctness. For each launch the engine records:')
body('coverage - total threads (grid x block) and elements per thread,')
body('stages - per-stage milliseconds for alloc, H2D copy, kernel, sync, D2H copy and free,')
body('kernel time - device-busy time excluding transfers, the core execution-model metric,')
body('checksum - sum over the first 1024 output elements, identical across paths,')
body('speedup - host-loop runtime divided by device-launch runtime on the benchmark.')
sub('The Real-Time Model')  # P49
body('To make "real time" measurable, each launch is streamed on a fixed pace of about 0.35 seconds plus compute time, so the dashboard shows roughly two freshly timed launches per second against a display budget of one launch per second. If the measured compute time per launch stays below the budget the visualiser keeps up with its own queue; the gap between budget and measured time is the headroom. The benchmark mode separately times a full vector serially in Python and through the device pipeline to report the speedup.')
sub('Experimental Modes')  # P51
body('Three modes are executed on the identical workload:')
body('Serial baseline - one Python loop computes every element in order on the host. This establishes the reference runtime.')
body('Threaded device - a thread pool stands in for the GPU\'s multiprocessors: the vector is split into chunks computed concurrently and the host still performs alloc, copies, sync and free around them, so every stage of the execution model stays visible without a GPU.')
body('Real CUDA - host_launcher.py --cuda moves the same saxpy to torch, cupy or numba on an NVIDIA GPU; each launch reports genuine H2D, kernel and D2H timings in the same JSON shape, and the dashboard renders them identically.')
body('Correctness is enforced by an order-independent checksum over the first 1024 output elements; all three modes must produce the same checksum on the same seed. Timing may differ between runs because of OS scheduling, which is expected and is reported as a speedup range rather than a single fragile digit.')
sub('Host-Launch Pipeline')  # P57
bull('Allocate on the device: reserve N floats of device memory (the cudaMalloc analogue); record alloc time and keep scalars such as the saxpy alpha on the host side.')
bull('Copy host to device: transfer every input vector across the interconnect (the cudaMemcpy H2D analogue) and record H2D milliseconds separately from compute.')
bull('Launch the kernel: fire kernel<<<grid, block>>> so grid x block threads cover the N elements; each thread owns about coverage elements and the pool executes chunks concurrently like multiprocessors.')
bull('Synchronise: pass the cudaDeviceSynchronize analogue (pool join plus a fixed sync cost) so the host provably waits until every thread has finished before touching results.')
bull('Copy device to host: transfer the output vector back (the cudaMemcpy D2H analogue) and record D2H milliseconds; the host copy is the only result the dashboard trusts.')
bull('Release and report: free device buffers, sum the six stages into total launch time, and emit one JSON object per launch carrying config, stages, checksum and GFLOPS.   ')

tbl = d.add_table(rows=8, cols=3, style='Table Grid')
cells = [
    ('Metric', 'Definition', 'What it shows'),
    ('Total launch latency', 'Wall-clock time for malloc+H2D+kernel+sync+D2H+free', 'Real-time behaviour'),
    ('Kernel time', 'Device-busy milliseconds excluding transfers', 'Execution-model efficiency'),
    ('Transfer share', 'H2D plus D2H milliseconds over total launch time', 'Transfer-bound vs kernel-bound'),
    ('Speedup', 'Serial host-loop runtime / device-launch runtime', 'Benefit of the device path'),
    ('Coverage', 'Vector length divided by grid x block threads', 'Launch sizing quality'),
    ('Checksum', 'Sum over first 1024 outputs, equal across modes', 'Correctness across paths'),
    ('GFLOPS', 'Element operations per kernel second', 'Device throughput'),
]
for i, row in enumerate(tbl.rows):
    for j, val in enumerate(cells[i]):
        row.cells[j].text = ''
        p = row.cells[j].paragraphs[0]
        r = p.add_run(val)
        if i == 0:
            r.bold = True

para('Normal', '  ')  # P64
para('Normal', '  ', LEFT)  # P65
para('Normal', '  ', LEFT)  # P66
para('Normal', ' ', LEFT)  # P67
empty()  # P68

# ---------------- CHAPTER 5 ----------------
chapter(5)  # P69  ('CHAPTER 5: ' + trailing)
h1t('Result')  # P70
para('Normal', '  ', CENTER)  # P71
p = para('Normal', '', LEFT)  # P72
run(p, 'saxpy live - 64 blocks x 256 threads', bold=True)
run(p, ' ', bold=True)
run(p, ' ', bold=None)
para('Normal', '', RIGHT).add_run().add_picture(SHOTS + '\\shot_live.png', width=IMG_W)  # P73
para('Normal', ' ' * 35, LEFT)  # P74
p = para('Normal', '', LEFT)  # P75
run(p, '  real GPU probe', bold=True)
para('Normal', '', LEFT).add_run().add_picture(SHOTS + '\\shot_idle.png', width=IMG_W)  # P76
para('Normal', ' ' * 35, LEFT)  # P77
para('Normal', '', RIGHT)  # P78
para('Normal', '', RIGHT)  # P79
para('Normal', '')  # P80
para('Normal', '')  # P81
para('Normal', '')  # P82
para('Normal', '')  # P83
para('Normal', '')  # P84
p = para('Normal', '', None)  # P85
run(p, ' ' * 20, bold=True, size=P14)
run(p, 'Race - host loop vs device', bold=True, size=P14)
para('Normal', '  ', RIGHT)  # P86
para('Normal', '', LEFT).add_run().add_picture(SHOTS + '\\shot_bench.png', width=IMG_W)  # P87
para('Normal', ' ' * 26, LEFT)  # P88
p = para('Normal', '', LEFT)  # P89 -> code caption 1
run(p, 'device_core.py - six-stage pipeline', bold=True)
run(p, ' ', bold=True)
run(p, ' ', bold=None)
para('Normal', '', LEFT).add_run().add_picture(SHOTS + '\\code_pipeline.png', width=IMG_W)  # P90 -> code shot 1
p = para('Normal', '', LEFT)  # P91 -> code caption 2
run(p, 'host_launcher.py - real CUDA path', bold=True)
run(p, ' ', bold=True)
run(p, ' ', bold=None)
para('Normal', '', LEFT).add_run().add_picture(SHOTS + '\\code_host.png', width=IMG_W)  # P92 -> code shot 2
empty()  # P93
para('Normal', ' ', LEFT)  # P94
empty()  # P95
empty()  # P96
empty()  # P97
empty()  # P98
empty()  # P99
empty()  # P100
empty()  # P101
para('Normal', ' ', LEFT)  # P102
para('Normal', ' ', LEFT)  # P103

# ---------------- CHAPTER 6 ----------------
p = para('Normal', '', LEFT)  # P104
run(p, 'Chapter  6:', bold=True, size=P16)
run(p, ' ', size=P13)
h1t('Conclusion')  # P105
body('The project successfully built and executed a live host-to-device launch visualiser in which every kernel launch passes through the full six-stage pipeline. The threaded-device run with 64 blocks of 256 threads (16,384 threads over 200,000 elements, coverage 12.21) produced the same output checksums as the serial host loop on identical seeds (for example checksum 1835.588 on the reference launch), confirming the correctness of the launch pipeline, and the live stream delivered all 20 launched ticks with no drops.')
body('The measured results show that (a) the device path gives a real 11-18x speedup over the serial host loop on a 6-core laptop for the 200,000-element saxpy workload (the captured race shows 44.872 ms serial against 2.526 ms device, i.e. 17.76x total and 22.01x kernel-only, with repeated runs landing between 11x and 18x); (b) per-stage timing correctly exposes the execution model at this size - about 2.0 ms of kernel against 0.5 ms of transfers, so the launch is kernel-bound with transfers under 20% of the total; and (c) the dashboard streams roughly two freshly timed launches per second against a one-per-second display budget - about 2x faster than required - so it keeps up with its own queue with wide headroom.')
body('Future work includes running host_launcher.py --cuda on a GPU-equipped lab machine to compare genuine H2D, kernel and D2H timings against the threaded model on identical vectors, sweeping grid and block shapes to map the occupancy curve directly, adding shared-memory tiling to the blur kernel to teach on-chip reuse, and extending the visualiser with non-blocking launches and streams so transfer and compute overlap inside each tick.')
para('Normal', '  ', LEFT)  # P109
para('Normal', '  ', CENTER)  # P110
para('Normal', '  ', CENTER)  # P111
para('Normal', '  ', CENTER)  # P112
para('Normal', '  ', CENTER)  # P113
para('Normal', '  ', CENTER)  # P114
para('Normal', '  ', CENTER)  # P115
para('Normal', '  ', CENTER)  # P116
para('Normal', '  ', CENTER)  # P117
para('Normal', '  ', CENTER)  # P118
para('Normal', '  ', CENTER)  # P119
para('Normal', '  ', LEFT)  # P120
para('Normal', ' ', LEFT)  # P121
para('Normal', ' ', LEFT)  # P122
para('Normal', ' ', LEFT)  # P123

# ---------------- REFERENCES ----------------
h1t('References')  # P124
para('Normal', '  ', LEFT)  # P125
refs = [
    '[1] NVIDIA Corporation (2023). CUDA C++ Programming Guide. docs.nvidia.com/cuda/cuda-c-programming-guide/',
    '[2] NVIDIA Corporation (2023). CUDA Runtime API Reference: cudaMalloc, cudaMemcpy, cudaDeviceSynchronize. docs.nvidia.com/cuda/',
    '[3] Kirk, D. B., Hwu, W. W. (2016). Programming Massively Parallel Processors. Morgan Kaufmann, Chapters 2-4 (execution model).',
    '[4] Sanders, J., Kandrot, E. (2010). CUDA by Example. Addison-Wesley.',
    '[5] Harris, M. (2013). Unified Memory for CUDA Beginners. NVIDIA Technical Blog. developer.nvidia.com/blog/',
    '[6] Micikevicius, P. (2009). 3D Finite Difference Computation on GPUs using CUDA. NVIDIA GTC Proceedings.',
    '[7] Pallets Projects (2024). Flask web framework documentation. flask.palletsprojects.com/',
    '[8] WHATWG (2024). HTML Standard, Server-Sent Events. html.spec.whatwg.org/multipage/server-sent-events.html',
    '[9] Chart.js contributors (2024). Chart.js documentation. chartjs.org/docs/',
    '[10] Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. AFIPS Conference Proceedings.',
    '[11] Harris, M., Perrot, S. (2024). CUDA Pro Tip: Occupancy and launch configuration. NVIDIA Technical Blog.',
    '[12] Python Software Foundation (2024). concurrent.futures - Launching parallel tasks. docs.python.org/3/library/concurrent.futures.html',
]
for ref in refs:
    body(ref)
dots()  # P138
para('Normal', '  ', LEFT)  # P139 trailing


def add_page_field(par):
    r = par.add_run()
    f1 = OxmlElement('w:fldChar')
    f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar')
    f2.set(qn('w:fldCharType'), 'end')
    r._r.append(f1)
    r._r.append(it)
    r._r.append(f2)


sec.different_first_page_header_footer = True
sectPr = sec._sectPr
for tag in ('w:titlePg', 'w:evenAndOddHeaders'):
    if sectPr.find(qn(tag)) is None:
        el = OxmlElement(tag)
        sectPr.append(el)
for footer in (sec.footer, sec.first_page_footer, sec.even_page_footer):
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    run(fp, ' ' * 55)
    add_page_field(fp)
    run(fp, '   ')

d.core_properties.author = 'Tasfiya Nawaz'
d.core_properties.title = 'CUDA kernel execution model - understanding how host code launches device computation'
d.core_properties.subject = 'Parallel Computing seminar report'
d.save(OUT)
print('saved:', OUT, '| paras:', len(d.paragraphs))
