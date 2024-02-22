import numpy as np

SNR_CURRENT = 3.1
NOISE_ukarcmin_CURRENT = 30
NGAL_CURRENT = 300000

def forecast_SNR(snr_current, noise_current, ngal_current, noise_forecast, ngal_forecast):
    return snr_current * noise_current/noise_forecast * np.sqrt(ngal_forecast/ngal_current)

Ngal_forecast = 1000000
noise_ukarcmin_forecast = 10

forecasted = forecast_SNR(SNR_CURRENT, NOISE_ukarcmin_CURRENT, NGAL_CURRENT,
                          noise_ukarcmin_forecast, Ngal_forecast)


print("\\newcommand{\\CURRENTNOISE}{%i}" % NOISE_ukarcmin_CURRENT)
print("\\newcommand{\\CURRENTNGAL}{%s}" % f'{NGAL_CURRENT:,}')
print("\\newcommand{\\NOISEFORECAST}{%i}" % noise_ukarcmin_forecast)
print("\\newcommand{\\NGALFORECAST}{%s}" % f'{Ngal_forecast:,}')
print("\\newcommand{\\SNRFORECASTEDSO}{%i}" % forecasted)