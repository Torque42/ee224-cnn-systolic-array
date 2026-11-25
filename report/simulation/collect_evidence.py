from pathlib import Path
import bisect
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'report'/'simulation'
FIG=ROOT/'report'/'figures'

def read_vcd(path):
    scope, names, changes=[], {}, {}
    time=0
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts:
            continue
        if parts[0]=='$scope':
            scope.append(parts[2])
        elif parts[0]=='$upscope':
            scope.pop()
        elif parts[0]=='$var':
            width, code, name=int(parts[2]), parts[3], parts[4]
            names['.'.join(scope+[name])]=(code, width)
        elif line.startswith('#'):
            time=int(line[1:])
        elif line[0] in 'bB':
            value, code=line[1:].split()
            changes.setdefault(code, []).append((time, value))
        elif line[0] in '01xXzZ':
            changes.setdefault(line[1:], []).append((time, line[0]))
    return names, changes

def signal(vcd, name, shift=0, width=None):
    names, changes=vcd
    code, original_width=names[name]
    width=width or original_width
    result=[]
    for t, bits in changes[code]:
        if any(c in bits.lower() for c in 'xz'):
            value='X'
        else:
            value=(int(bits, 2)>>shift)&((1<<width)-1)
            if width>1 and value&(1<<(width-1)):
                value-=1<<width
        t=t/1000
        if not result or value!=result[-1][1]:
            result.append((t, value))
    return result

def value_at(events, t):
    return events[bisect.bisect_right([e[0] for e in events], t)-1][1]

def draw(name, traces, start, end, markers):
    lines=[r'\documentclass[tikz,border=5pt]{standalone}',
             r'\begin{document}', r'\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily\scriptsize]']
    scale=14/(end-start)
    bottom=-0.65*len(traces)
    for t in range(int(start), int(end)+1, 10):
        x=(t-start)*scale
        lines+=[fr'\draw[gray!20] ({x:.4f},.5)--({x:.4f},{bottom:.3f});',
                  fr'\node[above,font=\tiny] at ({x:.4f},.5) {{{t}}};']
    lines.append(r'\node[left] at (0,.95) {Time (ns)};')
    for index, (label, events, digital) in enumerate(traces):
        y=-index*0.65
        lines.append(fr'\node[left] at (-.12,{y:.3f}) {{{label}}};')
        points=[start]+[t for t,v in events if start<t<end]+[end]
        for left,right in zip(points,points[1:]):
            value=value_at(events,left)
            x1,x2=(left-start)*scale,(right-start)*scale
            if digital:
                level=y+(0.18 if value==1 else-0.18)
                lines.append(fr'\draw[thick] ({x1:.4f},{level:.3f})--({x2:.4f},{level:.3f});')
                if left>start:
                    lines.append(fr'\draw[thick] ({x1:.4f},{y-.18:.3f})--({x1:.4f},{y+.18:.3f});')
            else:
                lines.append(fr'\draw[fill=blue!3] ({x1:.4f},{y-.20:.3f}) rectangle ({x2:.4f},{y+.20:.3f});')
                if x2-x1>0.32:
                    lines.append(fr'\node[font=\tiny] at ({(x1+x2)/2:.4f},{y:.3f}) {{{value}}};')
    for t,label in markers:
        x=(t-start)*scale
        lines+=[fr'\draw[red!65!black,dashed] ({x:.4f},.3)--({x:.4f},{bottom:.3f});',
                  fr'\node[below,font=\tiny,text=red!65!black] at ({x:.4f},{bottom:.3f}) {{{label}}};']
    lines+=[r'\end{tikzpicture}',r'\end{document}']
    (FIG/f'{name}.tex').write_text('\n'.join(lines)+'\n')

pe=read_vcd(ROOT/'tb_processing_element.vcd')
prefix='tb_processing_element.'
traces=[(name,signal(pe,prefix+name),name in ('clock','reset'))
        for name in ('clock','reset','Ain','Bin','A','B','C')]
for t,expected in [(35,12),(45,-23),(55,16361),(65,105),(75,105),(80,0)]:
    assert value_at(signal(pe,prefix+'C'),t)==expected
assert value_at(signal(pe,prefix+'C'),2595)==4096000
assert signal(pe,prefix+'C')[-1][1]==-4064000

draw('pe_waveform',traces,0,100,[(35,'first sample'),(80,'asynchronous reset')])

vcd=read_vcd(ROOT/'tb_systolic_array.vcd')
prefix='tb_systolic_array.driver.'
reset=signal(vcd,prefix+'reset')
starts=[t for t,v in reset if v==0 and t>0][:2]
results={}
for index,(k,seed) in enumerate(((4,1),(8,9))):
    a=[[((r*29+q*7+seed)%256)-128 for q in range(k)] for r in range(4)]
    b=[[((q*11+c*31+seed*3)%256)-128 for c in range(4)] for q in range(k)]
    expected=[[sum(a[r][q]*b[q][c] for q in range(k)) for c in range(4)] for r in range(4)]

    first=starts[index]+15
    last=first+10*(k+5)
    actual=[]
    for r in range(4):
        row=[]
        for c in range(4):
            events=signal(vcd,prefix+'C',(r*4+c)*24,24)
            row.append(value_at(events,last))
        actual.append(row)
    assert actual==expected
    bottom=signal(vcd,prefix+'C',15*24,24)
    assert value_at(bottom,last-10)!=expected[3][3]
    assert value_at(bottom,last)==expected[3][3]
    edges=[t for t,v in signal(vcd,prefix+'clock') if v==1 and first<=t<=last]
    assert len(edges)==k+6
    results[f'B{"i" if k==4 else "ii"}']={
        'first_sample_ns':first,'complete_ns':last,'active_edges':len(edges),
        'observed_matrix':actual,'expected_matrix':expected,'all_entries_match':True}
    traces=[('clock',signal(vcd,prefix+'clock'),True),('reset',reset,True)]
    for port,lane in [('Ain',0),('Ain',3),('Bin',0),('Bin',3)]:
        traces.append((f'{port} lane {lane}',signal(vcd,prefix+port,lane*8,8),False))
    for r,c in [(0,0),(0,3),(3,0),(3,3)]:
        traces.append((f'C({r},{c})',signal(vcd,prefix+'C',(r*4+c)*24,24),False))
    draw(f'array_k{k}_waveform',traces,starts[index]-10,last+10,
         [(first,'edge 1'),(last,f'edge {len(edges)}: complete')])
results['A']={'initial_sequence_verified':True,'positive_250_term_sum':4096000,
              'negative_250_term_sum':-4064000,'asynchronous_reset_verified':True}
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
