import scipy.io as sc
import numpy as np
import skimage as ski
import pandas as pd
import matplotlib.pyplot as plt
import os

def blur_deblur_func(image,R,C,direction,L=25,inv=0):

    """
    applying motion blur and inversion function
    with inv flag we inverse the motion blur
    """
    kernel = np.zeros((R, C))
    idx = np.arange(-(L // 2), L // 2 + 1)
    # Horizontal motion
    if(direction=="Horizontal"):
        kernel[R // 2,C // 2 - L//2 : C // 2 + L//2 + 1] = 1
    elif(direction=="Vertical"):
        kernel[R//2 - L//2 : R//2+L//2 +1 , C // 2 ] = 1

    elif direction == "Diagonal":
        kernel[R // 2 + idx, C // 2 + idx] = 1          # FIXED: a diagonal LINE, not an LxL square
    else:
      raise ValueError(f"Unknown direction: {direction}")
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    kernel_shifted = np.fft.ifftshift(kernel)

    H = np.fft.fft2(kernel_shifted)
    ##applying blur
    if(inv==0):
        G = np.fft.fft2(image)*H
        blurred_img = np.fft.ifft2(G)
        return blurred_img
    else:

       #inverse filtering
       return np.fft.ifft2((np.fft.fft2(image))/(H + 1e-5))
        ##applying inversion function
        ## 1e-4 is used to avoid division by 0 exceptions

def wiener_filter(img,kernel,lam=0.01):
    #wiener filter implementation
    G = np.fft.fft2(img)
    h = np.fft.fft2(kernel)
    #K_reg = 0.01  # Tune this regularization parameter
    H_conj = np.conj(h)
    H_abs_sq = np.abs(h)**2
    Wiener = H_conj / (H_abs_sq + lam)
    F_hat = Wiener * G
    restored = np.fft.ifft2(F_hat).real
    #Clip pixel values to valid range
    restored = np.clip(restored, 0, 1)
    return restored

def wiener_filter_std(img,kernel,S_n,S_f):
    #we can use this function if we know PSD of noise and clean image
    G = np.fft.fft2(img)
    h = np.fft.fft2(kernel)
    #K_reg = 0.01  # Tune this regularization parameter
    H_conj = np.conj(h)
    H_abs_sq = np.abs(h)**2
    Wiener = H_conj / (H_abs_sq + S_n/S_f)
    F_hat = Wiener * G
    restored = np.fft.ifft2(F_hat).real
    #Clip pixel values to valid range
    restored = np.clip(restored, 0, 1)
    return restored

def periodogram2d(img):
    F = np.fft.fft2(img)
    return np.abs(F)**2 / img.size



def performance_evaluation(original, restored, data_range=1.0):
    """
    Evaluate image restoration performance.

    Parameters
    ----------
    original : ndarray
        Original sharp image.
    restored : ndarray
        Restored image.
    data_range : float

    Returns
    -------
    dict
        PSNR, SSIM, and MSE.
    """
    original = np.asarray(original, dtype=np.float64)
    restored = np.asarray(restored, dtype=np.float64)

    mse = np.mean((original - restored) ** 2)

    psnr = ski.metrics.peak_signal_noise_ratio(
        original, restored, data_range=data_range
    )

    ssim = ski.metrics.structural_similarity(
        original, restored, data_range=data_range
    )

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MSE": mse
    }        


data = sc.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_restoration.mat"))
I1 = data["I1"].astype(float)/255.0
I2 = data["I2"].astype(float)/255.0
Images = [I1,I2]

## applying fast ft on image

for image in Images:

    print("shape of Fourier tranform of I1 : " + str(np.fft.fft2(image).shape))
    R,C = np.fft.fft2(image).shape


    blurred_img_v = blur_deblur_func(image,R,C,"Vertical",25,0)
    ##adding gaussian noise
    noisy_img = ski.util.random_noise(np.real(blurred_img_v),mode='gaussian',var=0.0009)

    #plots to show different blurs
    fig,axs = plt.subplots(1,5,figsize=(12,8))
    axs[1].imshow(np.real(blur_deblur_func(image,R,C,"Vertical",25)),cmap="gray")
    axs[1].set_title("Blurred image (vertical)")
    axs[0].imshow(image,cmap="gray")
    axs[0].set_title("Original image")
    axs[2].imshow(np.real(blur_deblur_func(image,R,C,"Horizontal",25)),cmap="gray")
    axs[2].set_title("Blurred image (horizontal)")
    axs[3].imshow(np.real(blur_deblur_func(image,R,C,"Diagonal",25)),cmap="gray")
    axs[3].set_title("Blurred image (diagonal)")
    axs[4].imshow(np.real(noisy_img),cmap="gray")
    axs[4].set_title("Vertical blurring + noise image")
    plt.suptitle("Different types of linear blurs")
    plt.tight_layout()
    plt.show()

    #plots for inversions of motion blurs
    blur_img_h = blur_deblur_func(image,R,C,"Horizontal",25,0)
    blur_img_d = blur_deblur_func(image,R,C,"Diagonal",25,0)
    restored_img_v = blur_deblur_func(blurred_img_v,R,C,"Vertical",25,1)
    restored_img_h = blur_deblur_func(blur_img_h,R,C,"Horizontal",25,1)
    restored_img_d = blur_deblur_func(blur_img_d,R,C,"Diagonal",25,1)
    fig,axs = plt.subplots(1,3,figsize=(12,8))
    axs[0].imshow(np.real(restored_img_v),cmap="gray")
    axs[0].set_title("Restored image (vertical)")
    axs[1].imshow(np.real(restored_img_h),cmap="gray")
    axs[1].set_title("Restored image (horizontal)")
    axs[2].imshow(np.real(restored_img_d),cmap="gray")
    axs[2].set_title("Restored image (diagonal)")
    plt.suptitle("Outputs of inversion filter")
    plt.tight_layout()
    plt.show()

    kernels= {}
    kernel = np.zeros((R,C))
    idx = np.arange(-(25 // 2), 25 // 2 + 1)
    kernel[R // 2,C // 2 - 25//2 : C // 2 + 25//2 + 1] = 1 #horizontal
    kernels['Horizontal'] = kernel
    kernel = np.zeros((R,C))
    kernel[R//2 - 25//2 : R//2+25//2 +1 , C // 2 ] = 1 #vertical
    kernels['Vertical'] = kernel
    kernel = np.zeros((R,C))
    kernel[R // 2 + idx, C // 2 + idx] = 1 #diagonal
    kernels['Diagonal'] = kernel
    #rev_k = {v: k for k, v in kernels.items()}
    for name,kernel in kernels.items():
        kernel = kernel / np.sum(kernel)
        kernel_shifted = np.fft.ifftshift(kernel)

        # adding gaussian noise with std. deviation of "sigma"
        sigma = 0.03
        np.random.seed(42)
        Sf = np.maximum(periodogram2d(image), 1e-12)
        Sn = sigma**2
        noise = np.random.normal(0, sigma, (R, C))
        noisy_img = blur_deblur_func(image,R,C,name,25,0) + noise

        #invoke wiener filter function call
        #wiener_filter(np.real(noisy_img),kernel_shifted)
        wiener_std= wiener_filter_std(np.real(noisy_img),kernel_shifted,Sn,Sf).real
        perf_metric_wiener_std_restored = performance_evaluation(image,wiener_std)
        print("The below are performance metrics when we know PSD of noise : ")
        print("PSNR is : " + str(perf_metric_wiener_std_restored['PSNR']))
        print("SSIM is : " + str(perf_metric_wiener_std_restored['SSIM']))
        print("MSE is : " + str(perf_metric_wiener_std_restored['MSE']))

        # Search for the optimal regularization parameter
        #lambdas = [1e-5,1e-4,1e-3,1e-2,1e-1,0.001,0.005,0.01,0.02,0.03,0.06]
        lambdas = np.logspace(-5, 0, 21)
        #lambdas = np.logspace(-5, 1, 50)
        print(lambdas)
        results = []

        for lam in lambdas:
            restored = wiener_filter(np.real(noisy_img),kernel_shifted,lam)

            metrics = performance_evaluation(
                image, restored, data_range=1.0
            )

            results.append({
                "lambda": lam,
                **metrics
            })

        results_df = pd.DataFrame(results)
        for index, row in results_df.iterrows():
            print(f"Result {index + 1}:")
            print(f"  lambda: {row['lambda']:.2e}")
            print(f"  PSNR: {row['PSNR']:.4f} dB")
            print(f"  SSIM: {row['SSIM']:.4f}")
            print(f"  MSE:  {row['MSE']:.4f}")
            print("-" * 30)

        # Find the best lambda based on PSNR peak-signal-to-noise-ratio
        best_idx = results_df["MSE"].idxmin()
        best_lambda_MSE = results_df.loc[best_idx, "lambda"]
        best_idx = results_df["SSIM"].idxmax()
        best_lambda_SSIM = results_df.loc[best_idx, "lambda"]
        best_idx = results_df["PSNR"].idxmax()
        best_lambda_PSNR = results_df.loc[best_idx, "lambda"]

        print(f"Best lambda: {best_lambda_MSE} for {name} blurring based on MSE is : ")
        print(results_df.loc[results_df["MSE"].idxmin(), "lambda"])
        print(f"Best lambda: {best_lambda_PSNR} for {name} blurring based on PSNR is : ")
        print(results_df.loc[results_df["PSNR"].idxmax(), "lambda"])
        print(f"Best lambda: {best_lambda_SSIM} for {name} blurring based on SSIM is : ")
        print(results_df.loc[results_df["SSIM"].idxmax(), "lambda"])

        # Plots for  performance metrics
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        for axis, metric in zip(ax, ["PSNR", "SSIM", "MSE"]):
            axis.semilogx(results_df["lambda"], results_df[metric])
            axis.set_xlabel("Regularization parameter (lambda)")
            axis.set_ylabel(metric)
            axis.set_title(f"{metric} vs Lambda")
            axis.grid(True)
        plt.suptitle(f"For {name} blurring")
        plt.tight_layout()
        plt.show()

        # Display the best restored image based on 
        #best_restored = wiener_filter(np.real(noisy_img),kernel_shifted, best_lambda)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(np.real(noisy_img), cmap="gray", vmin=0, vmax=1)
        plt.title("Blurred + Noisy")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(wiener_std, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Restored image for Sn/Sf")
        plt.axis("off")

        plt.suptitle(f"For {name} blurring")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(wiener_filter(np.real(noisy_img),kernel_shifted, best_lambda_MSE), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Restored image based on MSE")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(wiener_filter(np.real(noisy_img),kernel_shifted, best_lambda_PSNR), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Restored image based on PSNR")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(wiener_filter(np.real(noisy_img),kernel_shifted, best_lambda_SSIM), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Restored image based on SSIM")
        plt.axis("off")
        plt.suptitle("Outputs based on best lambda for varying metrics")
        plt.tight_layout()
        plt.show()

