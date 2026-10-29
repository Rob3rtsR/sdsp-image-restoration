Image restoration project (EE4795, group 7.3)

image_restoration.py: blurs the two test images (25-tap motion blur, horizontal, vertical and diagonal), adds Gaussian noise (sigma = 0.03) and restores them with the inverse filter and the Wiener filter (constant K over a range of lambda, and with Sn/Sf from the clean image). It prints PSNR, SSIM and MSE and plots the results. It opens a lot of plot windows, close each one to continue.

report_tables_figures.py: recreates Tables 1 and 2 and Figures 1 to 6 of the report with the same settings (seed 42). The output goes to the folder report_output.

Both scripts read img_restoration.mat from their own folder. They need numpy, scipy, scikit-image, pandas and matplotlib.

python image_restoration.py
python report_tables_figures.py
