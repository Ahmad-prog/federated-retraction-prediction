import sys; sys.path.insert(0, "/home/ahmad/.claude/skills/create-drawio")
from drawio import *

OUT = "/mnt/c/recearch/frederated_learning/paper/ipm/figures"

# ---------- Figure A: systematic study-selection flow (literature review) ----------
W, H = 150, 118
fig, ax = fig_ax(W, H, tw=10)
title(ax, 4, H-4, "Study selection protocol")

x0, bw, bh = 10, 60, 12
ex_x, ex_w = 92, 46
# main column boxes (bottom y)
S1=(x0,100,"Records identified in databases\nScopus · WoS · ACL · DBLP · arXiv","blue")
S2=(x0,82,"After duplicate removal","blue")
S3=(x0,64,"Title & abstract screening\n(2015-2026, on-topic)","purple")
S4=(x0,46,"Full-text eligibility\n(task in scope)","purple")
S5=(x0,28,"Open-artifact filter:\nopen data or code?","yellow")
S6=(x0,8,"Studies synthesised (+ snowballing)","blue")
for (x,y,t,c) in [S1,S2,S3,S4,S5,S6]:
    box(ax,x,y,bw,bh,t,c,fs=10)
# vertical arrows
for a,b in [(S1,S2),(S2,S3),(S3,S4),(S4,S5),(S5,S6)]:
    arrow(ax,(a[0]+bw/2,a[1]),(b[0]+bw/2,b[1]+bh))
# excluded side boxes
E1=(ex_x,82,"Duplicates removed","gray")
E2=(ex_x,64,"Off-topic / out of\nyear range","gray")
E3=(ex_x,46,"Task out of scope","gray")
CX=(ex_x,28,"Retained for context\n(no open artifact)","gray")
for (x,y,t,c) in [E1,E2,E3,CX]:
    box(ax,x,y,ex_w,bh,t,c,fs=9)
arrow(ax,(S2[0]+bw,S2[1]+bh/2),(E1[0],E1[1]+bh/2))
arrow(ax,(S3[0]+bw,S3[1]+bh/2),(E2[0],E2[1]+bh/2))
arrow(ax,(S4[0]+bw,S4[1]+bh/2),(E3[0],E3[1]+bh/2))
arrow(ax,(S5[0]+bw,S5[1]+bh/2),(CX[0],CX[1]+bh/2))
# reproducible core annotation (green) beside S6
box(ax, ex_x, 8, ex_w, bh, "Reproducible core: e.g.\nUsman & Balke; Fletcher & Stevenson","green",fs=9)
arrow(ax,(S6[0]+bw,S6[1]+bh/2),(ex_x,8+bh/2))
save(fig, f"{OUT}/review_flow.png")
print("saved review_flow.png")

# ---------- Figure B: corpus construction pipeline (methodology) ----------
W, H = 190, 52
fig, ax = fig_ax(W, H, tw=10)
title(ax, 4, H-5, "Corpus construction")
flow_row(ax, [
    ("Retraction Watch\n(via Crossref)","blue"),
    ("Research articles\nw/ resolvable DOI","purple"),
    ("OpenAlex\nenrichment","orange"),
    ("Publisher x year\nmatched controls 3:1","yellow"),
    ("Feature extraction\n15 metadata + ling.","green"),
    ("Non-IID silos\npublisher / discipline","blue"),
], y=22, x0=4, bw=28, bh=15, gap=3)
box(ax, 4, 5, 182, 11,
    "37,648 retracted  +  100,889 controls  =  138,537 articles   (~50k with open-access full text)",
    "gray", fs=10)
save(fig, f"{OUT}/corpus_pipeline.png")
print("saved corpus_pipeline.png")
