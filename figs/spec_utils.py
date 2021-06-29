import pdb
import numpy as np
from scipy.io import loadmat

def project_power(dat, proj_mat, roi_ind):
    chan_ind_vals = np.nonzero(proj_mat.mean(1)!=0)[0]
    return np.dot(proj_mat[chan_ind_vals,roi_ind], dat[chan_ind_vals,:])

def _calc_dens_norm_factor(elec_locs, headGrid, projectionParameter):
    '''Calculate the factors (scalar values, each for a electrode) that normalize the elecrode
    projected density inside brain volume (makes its sum to be equal to one).'''

    # create a parameter set but withough normalization
    newProjectionParemeter = projectionParameter.copy()
    newProjectionParemeter['normalizeInBrainDipoleDenisty'] =  False


    # get each dipole density from gaussianWeightMatrix
    projectionMatrix, totalDipoleDensity, gaussianWeightMatrix = _getProjectionMatrix(elec_locs, headGrid,
                                                                                     newProjectionParemeter,
                                                                                     headGrid['insideBrainCube'])

    # calculate its sum inside brain volume, 1/this gives the factor which should be
    # multiplied by electrode density in projection (because we assume electrode should be
    # somewhere in the brain volume).
    dipoleInBrainDensityNormalizationFactor = np.ones((gaussianWeightMatrix.shape[0])) / gaussianWeightMatrix.sum(1)
    return dipoleInBrainDensityNormalizationFactor


def _getProjectionMatrix(elec_locs, headGrid, projectionParameter, regionOfInterestCube=None):
    '''gaussianWeightMatrix is electrodes x (requested) grid points. It contains electrode density
    at each grid point for each electrode.
    
    Conversion of @bigdelys Matlab functions to Python'''
    if regionOfInterestCube is None:
        regionOfInterestCube = headGrid['insideBrainCube'].copy()

    if isinstance(regionOfInterestCube, str):
        if (regionOfInterestCube == 'all'):
            regionOfInterestCube = np.ones(headGrid['cubeSize'][0].astype('int'), dtype=bool)

    sd_est_err_pow2 = projectionParameter['sd_est_elecloc'] ** 2

    # projection matrix is number of electrodes x number of grid point inside brain volume
    n_pnts_roi = regionOfInterestCube.sum()

    n_elecs = elec_locs.shape[0]
    projectionMatrix = np.zeros((n_elecs, n_pnts_roi))
    totalDipoleDensity = np.zeros((n_pnts_roi))
    gaussianWeightMatrix = np.zeros((n_elecs, n_pnts_roi))
    dist_elec_gridlocs = np.zeros((n_elecs, n_pnts_roi))

    # swap axes to account for Matlab/Python differences in flattening 3D arrays
    if headGrid['xCube'].shape[0] > headGrid['xCube'].shape[2]:
        regionOfInterestCube = np.swapaxes(regionOfInterestCube, 0, 2)
        headGrid['xCube'] = np.swapaxes(headGrid['xCube'], 0, 2)
        headGrid['yCube'] = np.swapaxes(headGrid['yCube'], 0, 2)
        headGrid['zCube'] = np.swapaxes(headGrid['zCube'], 0, 2)
        headGrid['insideBrainCube'] = np.swapaxes(headGrid['insideBrainCube'], 0, 2)
    
    # a N x 3 matrix (N is the number of grid points inside brain volume
    gridPosition = np.vstack((headGrid['xCube'][regionOfInterestCube],
                              headGrid['yCube'][regionOfInterestCube],
                              headGrid['zCube'][regionOfInterestCube])).T

    if projectionParameter['normalizeInBrainDipoleDenisty']:
        dipoleInBrainDensityNormalizationFactor = _calc_dens_norm_factor(elec_locs, headGrid, projectionParameter)

    for dipoleNumber in range(n_elecs):
        # first place distance in the array
        dist_elec_gridlocs[dipoleNumber,:] = np.sum((gridPosition - np.tile(elec_locs[dipoleNumber, :],
                                                                            [gridPosition.shape[0], 1]))**2, 1)**0.5

        normalizationFactor = 1 / (projectionParameter['sd_est_elecloc']**3 * np.sqrt(8 * (np.pi**3)))
        gaussianWeightMatrix[dipoleNumber,:] = normalizationFactor * np.exp(-dist_elec_gridlocs[dipoleNumber,:]**2 / (2 * sd_est_err_pow2))

        # truncate the dipole denisty Gaussian at ~3 standard deviation
        gaussianWeightMatrix[dipoleNumber, dist_elec_gridlocs[dipoleNumber, :] > (projectionParameter['n_sd_trunc_gaussian'] *\
                                                                                  projectionParameter['sd_est_elecloc'])] = 0

        # normalize the dipole in-brain denisty (make it sum up to one)
        if projectionParameter['normalizeInBrainDipoleDenisty']:
            gaussianWeightMatrix[dipoleNumber, :] = gaussianWeightMatrix[dipoleNumber, :] *\
                                                    dipoleInBrainDensityNormalizationFactor[dipoleNumber]

    # normalize gaussian weights to have the sum of 1 at each grid location
    for gridId in range(gaussianWeightMatrix.shape[1]):
        totalDipoleDensity[gridId] = np.sum(gaussianWeightMatrix[:, gridId])
        if totalDipoleDensity[gridId] > 0:
            projectionMatrix[:, gridId] = gaussianWeightMatrix[:, gridId] / totalDipoleDensity[gridId]

    return projectionMatrix, totalDipoleDensity, gaussianWeightMatrix

def proj_mat_compute(elec_locs, hgrid_fid, fwhm=20, bad_chans=[], aal_fid=None):
    '''Compute projection matrix from electrodes to regions of interest
    fwhm : full width at half maximum (in mm)'''
    hgrid = loadmat(hgrid_fid, matlab_compatible=True)
    headGrid_in = hgrid['headGrid'][0,0]

    sd_est_elecloc = fwhm/2.355  # this calculates sigma in Gaussian equation
    proj_param = {}
    proj_param['sd_est_elecloc'] = sd_est_elecloc
    proj_param['n_sd_trunc_gaussian'] = 10*sd_est_elecloc
    proj_param['normalizeInBrainDipoleDenisty'] = True

    _, totalDipoleDensity, gaussianWeightMatrix = _getProjectionMatrix(elec_locs, headGrid_in, proj_param)

    # Remove bad electrodes by zeroing out their projection values
    if len(bad_chans) > 0:
        if len(bad_chans)==1:
            bad_chans = bad_chans[0]

        gaussianWeightMatrix[bad_chans, :] = 0
        sum_vals = gaussianWeightMatrix.sum(0)
        for s in range(len(sum_vals)):
            gaussianWeightMatrix[:,s] = gaussianWeightMatrix[:,s]/sum_vals[s]
    
    if aal_fid:
        aal_rois = loadmat(aal_fid, matlab_compatible=True)['aal_rois']
        n_rois = aal_rois.shape[1]
        n_elecs = gaussianWeightMatrix.shape[0]

        labels = []
        for i in range(n_rois):
            aal_rois[0,i]['membershipProbabilityCube'] = np.swapaxes(aal_rois[0,i]['membershipProbabilityCube'], 0, 2)
            labels.append(''.join(aal_rois[0,i]['label'][0]))

        # Compute projection matrix onto specific AAL regions
        dipoleProbabilityInRegion = np.zeros((n_elecs, n_rois))
        for i in range(n_rois):
            dipoleProbabilityInRegion[:, i] = gaussianWeightMatrix @ aal_rois[0,i]['membershipProbabilityCube'][headGrid_in['insideBrainCube']]
        dipoleDensityROI = dipoleProbabilityInRegion.sum(0)

        # Normalize across ROI's (necessary for scaling)
        normdipoleProbabilityInRegion = np.zeros((n_elecs, n_rois))
        for j in range(n_rois):
            normdipoleProbabilityInRegion[:,j] = dipoleProbabilityInRegion[:,j]/dipoleProbabilityInRegion[:,j].sum()

        return dipoleDensityROI, normdipoleProbabilityInRegion, labels
    
    return totalDipoleDensity, gaussianWeightMatrix