import numpy as np

SNR_CURRENT = 3.1
NOISE_ukarcmin_CURRENT = 14
NGAL_CURRENT =  227837

def forecast_SNR(snr_current, noise_current, ngal_current, noise_forecast, ngal_forecast):
    return snr_current * noise_current/noise_forecast * np.sqrt(ngal_forecast/ngal_current)

Ngal_forecast = 4000000
#noise_ukarcmin_forecast = 6 # SO forecast
noise_ukarcmin_forecast = 14 # SO forecast NO map noise dependence

forecasted = forecast_SNR(SNR_CURRENT, NOISE_ukarcmin_CURRENT, NGAL_CURRENT,
                          noise_ukarcmin_forecast, Ngal_forecast)


print("\\newcommand{\\CURRENTNOISE}{%i}" % NOISE_ukarcmin_CURRENT)
print("\\newcommand{\\CURRENTNGAL}{%s}" % f'{NGAL_CURRENT:,}')
print("\\newcommand{\\NOISEFORECAST}{%i}" % noise_ukarcmin_forecast)
Ngal_forecast_to_print_million = int(Ngal_forecast/1e6)
print("\\newcommand{\\NGALFORECASTMILLIONS}{%s}" % f'{Ngal_forecast_to_print_million:,d}')
print("\\newcommand{\\SNRFORECASTEDSO}{%i}" % forecasted)