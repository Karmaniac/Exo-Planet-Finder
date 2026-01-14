import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

def searchSector(lc):
    lc_clean = lc.remove_outliers(sigma=5)
    lc_flat = lc_clean.flatten(window_length=401)

    periods = np.linspace(0.5, 20, 10000)
    bls = BoxLeastSquares(lc_flat.time.value, lc_flat.flux.value)

    power = bls.power(periods, np.linspace(0.02, 0.3, 20))

    best_index = np.argmax(power.power)
    best_period = power.period[best_index]

    best_duration = power.duration[best_index]
    best_t0 = power.transit_time[best_index]
    best_depth = power.depth[best_index]

    print('orbital period - ' + str(best_period)) #hot jupiter = 1-5, super earth 5-20
    print('orbital time - ' + str(best_duration))
    print('first time it passed the star - ' + str(best_t0))
    print('fractional drop in brightness - ' + str(best_depth * 100) + '%')

    lc_folded = lc_flat.fold(period=best_period, epoch_time=best_t0)
    lc_folded.plot()

    if checkFoldedCurve(lc_folded):
        print(planetValidation(best_index, best_period, best_duration, best_t0, best_depth, lc_flat))

    plt.show()



def planetValidation(index, period, duration, t0, depth, lc_flat):

    time = lc_flat.time.value
    flux = lc_flat.flux.valu
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period        # (Where is each datapoint relative to the center of transit)
    in_transit = np.abs(phase) < duration / 2                           # Boolean mask to mark true if datapoint is inside the transit
                                                                        # /2 so that you're centered around the middle of transit,
    out_transit = np.abs(phase) > duration                              # Boolean mask to mark false if not in the transit
    transit_number = np.floor((time - t0) / period)                     # assign a number to mark what part of transit (0 = 1st transit, 1 = 2nd, 2 = 3rd)


    # Measure each transit independently, depths[] to store each transit depth
    depths = []
    for n in np.unique(transit_number):
        mask = transit_number == n

        # Skip weakly masked transits
        if np.sum(mask & in_transit) < 5:
            continue

        #measure transit depth, out_transit = baseline brightness, in_transit = dimmed brightness
        depth = np.median(flux[mask & out_transit]) - np.median(flux[mask & in_transit])
        depths.append(depth)
    depths = np.array(depths)

    std = depths.std() / depths.mean()

    if depths.mean() > 0.025:
        isPlanet = False
        return isPlanet

    print(std)
    if std > 0.3:
        print(std)
        return [False, 'Instrument Failure']
    elif 0.3 >= std > 0.2:
        std = 'suspicious'
    elif 0.2 >= std > 0.1:
        std = 'acceptable'
    elif 0.1 >= std > 0:
        std = 'likely'
    else:
        return -1


    planetRadius = np.sqrt(depths.mean())
    print("Mean depth:", depths.mean())
    print("Std depth:", depths.std())

    odd = depths[::2]
    even = depths[1::2]

    if abs(np.mean(odd) - np.mean(even)) / depths.mean() > 0.10:
        return [False, 'Most Likely a Binary Star']

    planetRadius = np.sqrt(depths.mean())

    return [float(planetRadius), depth, std, planetRadius]


def checkFoldedCurve(folded):
    return True


lc = lk.search_lightcurve("TIC 307210830", mission="TESS").download()

searchSector(lc)