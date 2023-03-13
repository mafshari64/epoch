# python  Emax_proton_comparison.py


# parameters which we need to change are: nx, ny and the loop numbers and finally the name of the images which we saved in angular distribution plot.
 

import sys     #exit the code at a specific line
import sdf
import numpy as np
import matplotlib.pyplot as plt    
import time
import matplotlib.animation as animation
#from IPython import display
from matplotlib.font_manager import FontProperties
fp = FontProperties('Symbola')
import warnings
warnings.filterwarnings("ignore")
#constants: n,ii,ij
from matplotlib.colors import LogNorm
from matplotlib import ticker
import matplotlib.patches as mpatches   # plot ellipse
from matplotlib import colors as mcolors  


##### read Emax 

#### IMPORTANT: FOR 10-100 nm targets, the files should be copied from superMUC to jureca. 

### gaussian pulse
 
# emittance_10nm  = np.loadtxt('./3J-pulselength_time_1-target-10nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-10nm.txt')[:, 6]
# emittance_50nm  = np.loadtxt('./3J-pulselength_time_1-target-50nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-50nm.txt')[:, 6]
# emittance_100nm = np.loadtxt('./3J-pulselength_time_1-target-100nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-100nm.txt')[:, 6]

Emax_400nm = np.loadtxt('./Emax_vs_time1.txt')[:, 1] # 5th column
# emittance_1um = np.loadtxt('./3J-pulselength_time_1-target-1um/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-1um.txt')[:, 6] # 5th column
# emittance_4um = np.loadtxt('./3J-pulselength_time_1-target-4um/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-4um.txt')[:, 6] # 5th column
# emittance_400nm_preplasma = np.loadtxt('./3J-pulselength_time_1-with-preplasma/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-400nm.txt')[:, 6] # 5th column
# emittance_400nm_7J = np.loadtxt('./7J-pulselength_time_1/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-400nm.txt')[:, 6] # 5th column

# emittance_10nm_col_log_10  = np.loadtxt('./3J-pulselength_time_1-target-10nm-collision_col_log_10/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-10nm-collision_col_log_10.txt')[:, 6]
Emax_400nm_col_log_10 = np.loadtxt('./3J-pulselength_time_1-target-400nm-collision_col_log_10-not-precise/Emax_vs_time1.txt')[:, 1] # 5th column
# emittance_1um_col_log_10   = np.loadtxt('./3J-pulselength_time_1-target-1um-collision_col_log_10/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-1um-collision_col_log_10.txt')[:, 6] # 5th column

Emax_10nm_col_log_auto  = np.loadtxt('./3J-pulselength_time_1-target-10nm-collision_col_log_auto/Emax_vs_time1.txt')[:, 1]
Emax_400nm_col_log_auto = np.loadtxt('./3J-pulselength_time_1-target-400nm-collision_col_log_auto/Emax_vs_time1.txt')[:, 1] # 5th column
Emax_400nm_col_ionize = np.loadtxt('./3J-pulselength_time_1-target-400nm-collisional-ionization/Emax_vs_time1.txt')[:, 1] # 5th column
Emax_1um_col_log_auto   = np.loadtxt('./3J-pulselength_time_1-target-1um-collision_col_log_auto/Emax_vs_time1.txt')[:, 1] # 5th column
Emax_4um_col_log_auto   = np.loadtxt('./3J-pulselength_time_1-target-4um-collision_col_log_auto/Emax_vs_time1.txt')[:, 1] # 5th column


### real pulse

Emax_10nm_col_log_auto_real_pulse_normal  = np.loadtxt('./real-laser-pulse-20220602-normal_shot-binary-collision-10nm/Emax_vs_time1.txt')[:, 1]
Emax_400nm_col_log_auto_real_pulse_normal = np.loadtxt('./real-laser-pulse-20220602-normal_shot-binary-collision-400nm/Emax_vs_time1.txt')[:, 1] # 5th column

Emax_400nm_col_ionize_real_pulse_normal   = np.loadtxt('./real-laser-pulse-20220602-normal_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 1] # 5th column
Emax_400nm_col_ionize_real_pulse_best   = np.loadtxt('./real-laser-pulse-20220608-best_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 1] # 5th column
Emax_400nm_col_ionize_real_pulse_bad   = np.loadtxt('./real-laser-pulse-20220512-bad_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 1] # 5th column

Emax_400nm_col_ionize_real_pulse_CI_BC   = np.loadtxt('./real-laser-pulse-20220608-best_shot-collision-binary-and-ionization-400nm/Emax_vs_time1.txt')[:, 1] # 5th column


# print ('Emax_400nm= ', Emax_400nm, ' (mm.mrad)')
# print ('Emax_10nm= ', Emax_10nm, ' (mm.mrad)')
# print (' ') 
 
# sys.exit()  

##### read time 

### gaussian pulse

# t_10nm  = np.loadtxt('./3J-pulselength_time_1-target-10nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-10nm.txt')[:, 0]
# t_50nm  = np.loadtxt('./3J-pulselength_time_1-target-50nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-50nm.txt')[:, 0]
# t_100nm = np.loadtxt('./3J-pulselength_time_1-target-100nm/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-100nm.txt')[:, 0]
t_400nm = np.loadtxt('./Emax_vs_time1.txt')[:, 0] # 5th column
# t_400nm_preplasma = np.loadtxt('./3J-pulselength_time_1-with-preplasma/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-400nm.txt')[:, 0] # 5th column
# t_400nm_7J = np.loadtxt('./7J-pulselength_time_1/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-400nm.txt')[:, 0] # 5th column
# t_4um = np.loadtxt('./3J-pulselength_time_1-target-4um/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-4um.txt')[:, 0] # 5th column
# t_1um = np.loadtxt('./3J-pulselength_time_1-target-1um/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-1um.txt')[:, 0] # 5th column

# t_1um_col_log_10 = np.loadtxt('./3J-pulselength_time_1-target-1um-collision_col_log_10/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-1um-collision_col_log_10.txt')[:, 0] # 5th column
t_400nm_col_log_10 = np.loadtxt('./3J-pulselength_time_1-target-400nm-collision_col_log_10-not-precise/Emax_vs_time1.txt')[:, 0] # 5th column
# t_10nm_col_log_10  = np.loadtxt('./3J-pulselength_time_1-target-10nm-collision_col_log_10/Emittance-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-10nm-collision_col_log_10.txt')[:, 0]

t_4um_col_log_auto = np.loadtxt('./3J-pulselength_time_1-target-4um-collision_col_log_auto/Emax_vs_time1.txt')[:, 0] # 5th column
t_1um_col_log_auto = np.loadtxt('./3J-pulselength_time_1-target-1um-collision_col_log_auto/Emax_vs_time1.txt')[:, 0] # 5th column
t_400nm_col_log_auto = np.loadtxt('./3J-pulselength_time_1-target-400nm-collision_col_log_auto/Emax_vs_time1.txt')[:, 0] # 5th column
t_400nm_col_ionize = np.loadtxt('./3J-pulselength_time_1-target-400nm-collisional-ionization/Emax_vs_time1.txt')[:, 0] # 5th column
t_10nm_col_log_auto  = np.loadtxt('./3J-pulselength_time_1-target-10nm-collision_col_log_auto/Emax_vs_time1.txt')[:, 0]

### real pulse
t_10nm_col_log_auto_real_pulse_normal  = np.loadtxt('./real-laser-pulse-20220602-normal_shot-binary-collision-10nm/Emax_vs_time1.txt')[:, 0]
t_400nm_col_log_auto_real_pulse_normal = np.loadtxt('./real-laser-pulse-20220602-normal_shot-binary-collision-400nm/Emax_vs_time1.txt')[:, 0] # 5th column

t_400nm_col_ionize_real_pulse_normal   = np.loadtxt('./real-laser-pulse-20220602-normal_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 0] # 5th column
t_400nm_col_ionize_real_pulse_best   = np.loadtxt('./real-laser-pulse-20220608-best_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 0] # 5th column
t_400nm_col_ionize_real_pulse_bad   = np.loadtxt('./real-laser-pulse-20220512-bad_shot-collisional-ionization-400nm/Emax_vs_time1.txt')[:, 0] # 5th column

t_400nm_col_ionize_real_pulse_CI_BC  = np.loadtxt('./real-laser-pulse-20220608-best_shot-collision-binary-and-ionization-400nm/Emax_vs_time1.txt')[:, 0] # 5th column

# sys.exit()  

########### select non zero elements

### time     first we should select time vs Emax (e.g.  t_10nm[Emax_10nm > 0] ) since if we first select Emax vs time, len(time) and len(Emax) will be different.

# t_10nm  = t_10nm[Emax_10nm > 0]
# t_10nm_col_log_10  = t_10nm_col_log_10[Emax_10nm_col_log_10 > 0]
t_10nm_col_log_auto  = t_10nm_col_log_auto[Emax_10nm_col_log_auto > 0]


# t_50nm  = t_50nm[Emax_50nm > 0]
# t_100nm  = t_100nm[Emax_100nm > 0]

t_400nm  = t_400nm[Emax_400nm > 0]
t_400nm_col_log_10  = t_400nm_col_log_10[Emax_400nm_col_log_10 > 0]
t_400nm_col_log_auto  = t_400nm_col_log_auto[Emax_400nm_col_log_auto > 0]
t_400nm_col_ionize  = t_400nm_col_ionize[Emax_400nm_col_ionize > 0]
# t_400nm_preplasma  = t_400nm_preplasma[Emax_400nm_preplasma > 0]
# t_400nm_7J  = t_400nm_7J[Emax_400nm_7J > 0]

# t_1um  = t_1um[Emax_1um > 0]
# t_1um_col_log_10  = t_1um_col_log_10[Emax_1um_col_log_10 > 0]
t_1um_col_log_auto  = t_1um_col_log_auto[Emax_1um_col_log_auto > 0]

# t_4um  = t_4um[Emax_4um > 0]
t_4um_col_log_auto  = t_4um_col_log_auto[Emax_4um_col_log_auto > 0]

### real pulse
t_10nm_col_log_auto_real_pulse_normal  = t_10nm_col_log_auto_real_pulse_normal[ Emax_10nm_col_log_auto_real_pulse_normal> 0]
t_400nm_col_log_auto_real_pulse_normal = t_400nm_col_log_auto_real_pulse_normal[ Emax_400nm_col_log_auto_real_pulse_normal> 0]

t_400nm_col_ionize_real_pulse_normal   = t_400nm_col_ionize_real_pulse_normal[ Emax_400nm_col_ionize_real_pulse_normal> 0]
t_400nm_col_ionize_real_pulse_best   = t_400nm_col_ionize_real_pulse_best[ Emax_400nm_col_ionize_real_pulse_best> 0]
t_400nm_col_ionize_real_pulse_bad   = t_400nm_col_ionize_real_pulse_bad[ Emax_400nm_col_ionize_real_pulse_bad > 0]
t_400nm_col_ionize_real_pulse_CI_BC   = t_400nm_col_ionize_real_pulse_CI_BC[ Emax_400nm_col_ionize_real_pulse_CI_BC > 0]


### Emax

# Emax_10nm  = Emax_10nm[Emax_10nm > 0]
# Emax_10nm_col_log_10  = Emax_10nm_col_log_10[Emax_10nm_col_log_10 > 0]
Emax_10nm_col_log_auto  = Emax_10nm_col_log_auto[Emax_10nm_col_log_auto > 0]

# Emax_50nm  = Emax_50nm[Emax_50nm > 0]
# Emax_100nm  = Emax_100nm[Emax_100nm > 0]

Emax_400nm  = Emax_400nm[Emax_400nm > 0]
Emax_400nm_col_log_10  = Emax_400nm_col_log_10[Emax_400nm_col_log_10 > 0]
Emax_400nm_col_log_auto  = Emax_400nm_col_log_auto[Emax_400nm_col_log_auto > 0]
Emax_400nm_col_ionize  = Emax_400nm_col_ionize[Emax_400nm_col_ionize > 0]
# Emax_400nm_preplasma  = Emax_400nm_preplasma[Emax_400nm_preplasma > 0]
# Emax_400nm_7J  = Emax_400nm_7J[Emax_400nm_7J > 0]

# Emax_1um  = Emax_1um[Emax_1um > 0]
# Emax_1um_col_log_10  = Emax_1um_col_log_10[Emax_1um_col_log_10 > 0]
Emax_1um_col_log_auto  = Emax_1um_col_log_auto[Emax_1um_col_log_auto > 0]

# Emax_4um  = Emax_4um[Emax_4um > 0]
Emax_4um_col_log_auto  = Emax_4um_col_log_auto[Emax_4um_col_log_auto > 0]
# print  ('Emax_10nm=',Emax_10nm)

### real pulse
Emax_10nm_col_log_auto_real_pulse_normal  = Emax_10nm_col_log_auto_real_pulse_normal[ Emax_10nm_col_log_auto_real_pulse_normal> 0]
Emax_400nm_col_log_auto_real_pulse_normal = Emax_400nm_col_log_auto_real_pulse_normal[ Emax_400nm_col_log_auto_real_pulse_normal> 0]

Emax_400nm_col_ionize_real_pulse_normal   = Emax_400nm_col_ionize_real_pulse_normal[ Emax_400nm_col_ionize_real_pulse_normal> 0]
Emax_400nm_col_ionize_real_pulse_best   = Emax_400nm_col_ionize_real_pulse_best[ Emax_400nm_col_ionize_real_pulse_best> 0]
Emax_400nm_col_ionize_real_pulse_bad   = Emax_400nm_col_ionize_real_pulse_bad[ Emax_400nm_col_ionize_real_pulse_bad > 0]
Emax_400nm_col_ionize_real_pulse_CI_BC   = Emax_400nm_col_ionize_real_pulse_CI_BC[ Emax_400nm_col_ionize_real_pulse_CI_BC > 0]




# sys.exit()

###################### 

plt.rc('font', family='serif')
 # plt.rc('text', usetex=True)
plt.rc('xtick' , labelsize=16)
plt.rc('ytick', labelsize=16)

######################  Emax

fig= plt.figure(1)
# ax = plt.subplot(111)

##### effect of collision of emmitance values

# fig= plt.figure(2)
# ax = plt.subplot(111)

# plt.plot( t_1um ,Emax_1um  ,  '-^c', label='1um') # cross line frm 
# plt.plot( t_4um ,Emax_4um  ,  '-sr', label='4um') # cross line frm 
# plt.plot( t_10nm ,Emax_10nm  ,  '-ok', label='10nm') # cross line frm 

# plt.plot( t_1um_col_log_10 ,Emax_1um_col_log_10  ,  '-.^c', label='1um_col_log_10') # cross line frm 
# plt.plot( t_10nm_col_log_10 ,Emax_10nm_col_log_10  ,  '-.ok', label='10nm_col_log_10') # cross line frm 

plt.plot( t_4um_col_log_auto ,Emax_4um_col_log_auto  ,  ':sr', label='4um_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 
plt.plot( t_1um_col_log_auto ,Emax_1um_col_log_auto  ,  ':<c', label='1um_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 

plt.plot( t_400nm_col_log_10 ,Emax_400nm_col_log_10  ,  '-.^y', label='400nm_gauss_FI+BC_log_10') # cross line frm 
plt.plot( t_400nm_col_log_auto ,Emax_400nm_col_log_auto  ,  ':^m', label='400nm_gauss_FI+BC_log_auto', markerfacecolor='none') # ,linewidth=7
plt.plot( t_400nm ,Emax_400nm  ,  '-db', label='400nm_gauss-FI', markerfacecolor='none') # cross line frm 
plt.plot( t_400nm_col_ionize ,Emax_400nm_col_ionize  ,  '--xg', label='400nm_gauss_FI+CI') # ,linewidth=7

plt.plot( t_10nm_col_log_auto ,Emax_10nm_col_log_auto  ,  ':ok', label='10nm_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 



plt.legend(loc='lower right', fontsize=8)

plt.title('   FI(field ionization); BC(binary collision); CI(collision ionization)', fontsize=13) #,color='green',  fontweight='bold')
   
# plt.xlabel('time (fs)', fontsize=15  ) # fontweight =  'ultralight', fontname ='Helvetica'
# plt.ylabel('$ \\epsilon $ ( mm-mrad)' , fontsize=15 )
plt.ylabel('E$ _{max} $ (MeV)' , fontsize=15 )

plt.xlim((0,1200))
plt.ylim((0,40))

# plt.savefig('Emax-' + str(E_threshold)+ 'Emax-'+ str(angle)+'degree-col_log_10_vs_auto.jpg', bbox_inches='tight')
plt.savefig('Emax-col_log_auto.jpg', bbox_inches='tight')

# plt.show()

# plt.pause(.5)
# plt.clf()

# sys.exit()

#### real pulse vs gauss pulse 
fig= plt.figure(2)

### gauss pulse- field ionization

# plt.plot( t_400nm ,Emax_400nm  ,  '-dm', label='400nm-FI') # cross line frm 

### gauss pulse- binary collision
# plt.plot( t_10nm_col_log_auto ,Emax_10nm_col_log_auto  ,  '-ok', label='10nm_gauss-BC_log_auto') # cross line frm 

# plt.plot( t_400nm_col_log_auto ,Emax_400nm_col_log_auto  ,  ':*g', label='400nm_gauss-BC_log_auto') # ,linewidth=7
# plt.plot( t_400nm_col_log_10 ,Emax_400nm_col_log_10  ,  ':Xg', label='400nm_gauss_BC_log_10') # cross line frm 

### gauss pulse- collision ionization
# plt.plot( t_400nm_col_ionize ,Emax_400nm_col_ionize  ,  ':dg', label='400nm_CI') # ,linewidth=7

# plt.plot( t_4um_col_log_auto ,Emax_4um_col_log_auto  ,  ':sr', label='4um_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 
# plt.plot( t_1um_col_log_auto ,Emax_1um_col_log_auto  ,  ':<c', label='1um_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 

plt.plot( t_400nm_col_log_auto ,Emax_400nm_col_log_auto  ,  ':^m', label='400nm_gauss_FI+BC_log_auto', markerfacecolor='none') # ,linewidth=7
# plt.plot( t_400nm_col_log_10 ,Emax_400nm_col_log_10  ,  '-.^g', label='400nm_gauss_FI+BC_log_10') # cross line frm 
plt.plot( t_400nm_col_ionize ,Emax_400nm_col_ionize  ,  '--xg', label='400nm_gauss_FI+CI', markerfacecolor='none') # ,linewidth=7
plt.plot( t_400nm ,Emax_400nm  ,  '-db', label='400nm_gauss-FI', markerfacecolor='none') # cross line frm 

plt.plot( t_10nm_col_log_auto ,Emax_10nm_col_log_auto  ,  '-.ok', label='10nm_gauss_FI+BC_log_auto', markerfacecolor='none') # cross line frm 



# real pulse- binary collision
plt.plot( t_10nm_col_log_auto_real_pulse_normal  ,Emax_10nm_col_log_auto_real_pulse_normal  , color='k', ls= '-.' ,  marker='o', label='10nm_real (normal)-FI+BC_log_auto') #, markerfacecolor='none' cross line frm 
plt.plot( t_400nm_col_log_auto_real_pulse_normal ,Emax_400nm_col_log_auto_real_pulse_normal  , color= 'm', ls= ':', marker='^', label='400nm_real (normal)-FI+BC_log_auto', markerfacecolor='none') # ,linewidth=7

# real pulse-  collision ionization
plt.plot( t_400nm_col_ionize_real_pulse_normal  ,Emax_400nm_col_ionize_real_pulse_normal , color= 'g' , ls= '--',marker='x', label='400nm_real (normal)-FI+CI') # cross line frm 
plt.plot( t_400nm_col_ionize_real_pulse_best ,Emax_400nm_col_ionize_real_pulse_best   , color= 'g' ,ls='--', marker='s', label='400nm_real (best)-FI+CI', markerfacecolor='none') # ,linewidth=7
plt.plot( t_400nm_col_ionize_real_pulse_CI_BC ,Emax_400nm_col_ionize_real_pulse_CI_BC , color= 'lime' ,  ls='--', marker='s', label='400nm_real (best)-FI+CI+BC', markerfacecolor='none') # ,linewidth=7
plt.plot( t_400nm_col_ionize_real_pulse_bad ,Emax_400nm_col_ionize_real_pulse_bad , color= 'g' ,  ls='--', marker='*', label='400nm_real (bad)-FI+CI') # ,linewidth=7

plt.vlines(x= 78, ymin=0, ymax=2, color='m', label='laser peak pulse_Gauss') # laser pulse has peak at 1000 fs + it takes 33.3 fs to reach target (xmin= -10 um)
plt.vlines(x= 1033.3, ymin=0, ymax=2, color='m', label='laser peak pulse_real') # laser pulse has peak at 1000 fs + it takes 33.3 fs to reach target (xmin= -10 um)

plt.legend(loc='lower right', fontsize=7,ncol=2) #  

# plt.title(' E$ _{max} $; Real pulse; \n  FI(field ionization); BC(binary collision); CI(collision ionization)', fontsize=10) #,color='green',  fontweight='bold')
plt.title(' FI(field ionization); BC(binary collision); CI(collision ionization)', fontsize=10) #,color='green',  fontweight='bold')
   
plt.xlabel('time (fs)', fontsize=15  ) # fontweight =  'ultralight', fontname ='Helvetica'
plt.ylabel('E$ _{max} $ (MeV)' , fontsize=15 )

plt.xlim((0,2800))
# plt.ylim((0,35))
plt.ylim((0,45))



plt.savefig('Emax-real-pulse-vs-gauss.jpg', bbox_inches='tight')

plt.show()

# plt.pause(.5)
plt.clf()

sys.exit()

