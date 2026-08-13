load ../../data/raw/start2_refmac1.pdb, aldc
hide everything
show cartoon, polymer
select active_site, chain A and (resi 222+224+235+281+301+401)
show sticks, chain A and (resi 222+224+235+281+401)
show spheres, chain A and resn ZN and resi 301
color gray70, elem C
color marine, elem N
color red, elem O
color slate, resn ZN
set sphere_scale, 0.45, resn ZN
distance zn_his222_ne2, chain A and resn ZN and resi 301, chain A and resn HIS and resi 222 and name NE2
distance zn_his224_ne2, chain A and resn ZN and resi 301, chain A and resn HIS and resi 224 and name NE2
distance zn_his235_nd1, chain A and resn ZN and resi 301, chain A and resn HIS and resi 235 and name ND1
distance zn_glu281_oe1, chain A and resn ZN and resi 301, chain A and resn GLU and resi 281 and name OE1
distance zn_edo401_o1, chain A and resn ZN and resi 301, chain A and resn EDO and resi 401 and name O1
distance zn_edo401_o2, chain A and resn ZN and resi 301, chain A and resn EDO and resi 401 and name O2
set dash_color, yellow
set dash_width, 2.5
zoom active_site, 8
