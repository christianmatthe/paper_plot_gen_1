import os

#Run all the plotting scripts
print("This may take several minutes")
filename = ("2023-04-21_1sccm_15A_TC_z-scan"
            +"_jf_wire_extract_recalib_just_plot.py")
print(f"Running {filename}")
os.system(f"python {filename}")

filename = ("Tschersich_shape_2024-04-17_paper_fBB_heatshield.py")
print(f"Running {filename}")
os.system(f"python {filename}")

filename = ("calib_analysis.py")
print(f"Running {filename}")
os.system(f"python {filename}")