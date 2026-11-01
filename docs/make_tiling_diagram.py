import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
W,H,T=1456,1088,640
cols=['#0F6FB0','#B84E08','#2E8B57']
fig,(a,b)=plt.subplots(2,1,figsize=(11,11),gridspec_kw={'height_ratios':[2.3,1.6]})
# A: frame with the top row of tiles (outline only), second row noted
a.add_patch(Rectangle((0,0),W,H,fc='#F2F5F9',ec='k',lw=2))
a.add_patch(Rectangle((512,0),128,T,fc='#F2A200',alpha=0.55,ec='none'))
for i,x in enumerate([0,512,816]):
    a.add_patch(Rectangle((x,0),T,T,fc='none',ec=cols[i],lw=3,ls='-' if i!=1 else '--'))
    a.text(x+16,24,f'Tile {i+1}',color=cols[i],fontsize=15,fontweight='bold',va='top')
for x in [0,512,816]:
    a.add_patch(Rectangle((x,448),T,T,fc='none',ec='#999999',lw=1.2,ls=':'))
a.text(728,860,'second row of 3 tiles (dotted), starts at y = 448',ha='center',fontsize=12,color='#555555')
a.add_patch(Rectangle((560,250),70,190,fc='none',ec='#C0392B',lw=3))
a.text(595,236,'person at the edge of Tile 1',ha='center',va='bottom',fontsize=12,color='#C0392B',fontweight='bold')
a.text(576,500,'shared strip',ha='center',fontsize=12,color='#8a5a00',fontweight='bold',rotation=0)
a.set_xlim(-40,W+40); a.set_ylim(H+40,-40); a.set_aspect('equal')
a.set_title('1456×1088 frame, cut (not resized) into 640×640 tiles: 3 across × 2 down = 6 tiles',fontsize=14,pad=10)
a.set_xticks([0,512,640,816,1456]); a.set_yticks([0,448,640,1088]); a.tick_params(labelsize=11)
# B: side view of the three tiles in one row
b.set_xlim(-40,W+40); b.set_ylim(-0.6,4.4); b.axis('off')
rows=[(0,3.3,'Tile 1: x = 0 to 640'),(512,2.2,'Tile 2: x = 512 to 1152'),(816,1.1,'Tile 3: x = 816 to 1456')]
for i,(x,y,t) in enumerate(rows):
    b.add_patch(Rectangle((x,y),T,0.75,fc=cols[i],alpha=0.28,ec=cols[i],lw=2,ls='--' if i==1 else '-'))
    b.text(x+T/2,y+0.37,t,ha='center',va='center',fontsize=13)
b.add_patch(Rectangle((512,2.2),128,1.85,fc='#F2A200',alpha=0.55,ec='none'))
b.add_patch(Rectangle((560,3.38),70,0.6,fc='none',ec='#C0392B',lw=2.5))
b.text(576,4.15,'overlap = 128 px (20% of 640)',ha='center',fontsize=12,color='#8a5a00',fontweight='bold')
b.text(1000,3.65,'Step between tiles = 640 − 128 = 512 px.\nTile 3 is shifted left to end at the frame edge,\nso its overlap with Tile 2 is larger (336 px).',fontsize=12,va='center')
b.text(0,0.35,'Each tile goes through the model separately; boxes are then merged back into frame\ncoordinates and duplicates from the shared strips are removed with NMS.',fontsize=12.5,va='center')
fig.tight_layout()
fig.savefig('tiles2.png',dpi=110)
