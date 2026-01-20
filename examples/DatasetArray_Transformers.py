#%% md
# # Transformers and Pipelines test on DatasetArray object
# 
# In this notebook we check the `caits.transformers` and Sklearn Pipelines consisting of `caits.transformers`.
# 
#%% md
# ## Importing libraries
#%%
import pandas as pd
from caits.filtering import filter_butterworth
from caits.fe import mean_value, std_value, stft, istft, melspectrogram
from caits.dataset import CoreArray, DatasetArray
from caits.transformers import (
    FunctionTransformer,
    FunctionTransformer2D,
    FeatureExtractorSignal,
    FeatureExtractorSpectrum,
    FeatureExtractorScalar,
    SlidingWindow
)
#%% md
# ## Dataset loading
# 
# For this notebook, we will use the data/AirQuality.csv dataset.
#%%
data = pd.read_csv("data/AirQuality.csv", sep=";", decimal=",")
data_X = data.iloc[:, 2:-4]
data_X = data_X.fillna(data_X.mean())
data_y = data.iloc[:, -4:-2]
data_y = data_y.fillna(data_y.mean())
#%%
data_X_vals = data_X.values
data_X_axis_names = {"axis_1": data_X.columns}
data_y_vals = data_y.values
data_y_axis_names = {"axis_1": data_y.columns}
data_X = CoreArray(values=data_X_vals, axis_names=data_X_axis_names)
data_y = CoreArray(values=data_y_vals, axis_names=data_y_axis_names)
datasetArrayObj = DatasetArray(data_X, data_y)
#%%
datasetArrayObj.X
#%% md
# ## FunctionTransformer
# 
# This transformer is mainly used for transforming the `X` attribute of the `DatasetArray` object into a `CaitsArray`s with the shape maintained.
# 
# We test the `caits.transformer.FunctionTransformer` using the `caits.fe.filter_butterworth` function.
# 
#%%
functionTransformer = FunctionTransformer(
    filter_butterworth,
    to_X=False,
    to_y=True,
    fs=200,
    filter_type='lowpass',
    cutoff_freq=50
)
transformedArray = functionTransformer.fit_transform(datasetArrayObj)
#%%
datasetArrayObj.X
#%%
transformedArray.X
#%%
datasetArrayObj.y
#%%
transformedArray.y
#%% md
# # FeatureExtractor
# 
# This transformer is mainly used for extracting single values per column or per row (if axis=1) for each instance of `DatasetArray.X`.
# 
# We test the `caits.transformer.FeatureExtractor` using the `caits.fe.mean_value` and `caits.fe.std_value`.
#%%
featureExtractor = FeatureExtractorScalar([
    {
        "func": mean_value,
        "params": {}
    },
    {
        "func": std_value,
        "params": {
            "ddof": 0
        }
    }
],
to_X=True,
to_y=True)
#%%
tmp = featureExtractor.fit_transform(datasetArrayObj)
tmp
#%%
tmp.X
#%%
tmp.y
#%% md
# ## FeatureExtractor2D
# 
# This transformer is mainly used for extracting 2D features per column of `DatasetArray.X`.
# 
# We test this using the `caits.fe.melspectrogram` and `caits.fe.stft`.
# Applying each of these functions will transform the `CaitsArray` of `DatasetArray.X` into a 3D `CaitsArray`.
# 
#%%
featureExtractor2D = FeatureExtractorSpectrum(
    melspectrogram,
    to_X=True,
    to_y=True,
    n_fft=100,
    hop_length=10
)
tmp = featureExtractor2D.fit_transform(datasetArrayObj)
#%%
tmp.X.shape
#%%
tmp.y.shape
#%%
featureExtractor2D = FeatureExtractorSpectrum(
    stft,
    to_X=True,
    to_y=True,
    n_fft=100,
    hop_length=10
)
tmp1 = featureExtractor2D.fit_transform(datasetArrayObj)
#%%
tmp1.X.iloc[:, 0, 0]
#%%
tmp1.y.iloc[:, 0, 0]
#%% md
# ## FunctionTransformer2D
# 
# This is mainly used to inverse the `featureExtractor2D` process. So, if `DatasetList.X` is a `CaitsArray` object, it will be
# transformed in a `CaitsArray`.
# 
# To test this we use the `caits.fe.istft` on the transformed `DatasetArray` object using `caits.fe.stft`.
#%%
functionTransformer = FunctionTransformer2D(
    istft,
    to_X=True,
    to_y=True,
    n_fft=100,
    hop_length=10
)
tmp2 = functionTransformer.fit_transform(tmp1)
#%%
tmp2.X
#%%
tmp2.y
#%% md
# ## SlidingWindow
# 
# This is used for performing the sliding window process in each instance of the `DatasetArray` object.
# 
# The final windows will be appended in a single `DatasetList` object.
#%%
slidingWindow = SlidingWindow(window_size=20, overlap=5)
tmp = slidingWindow.fit_transform(datasetArrayObj)
#%%
tmp
#%%
tmp.X
#%%
tmp.y
#%% md
# ## SklearnWrapper
#%%
from sklearn.preprocessing import StandardScaler
from caits.transformers import SklearnPipeStep, DatasetToArray, ArrayToDataset

dataFlatten = DatasetToArray(flatten=True, to_X=True, to_y=True)
scaler = SklearnPipeStep(StandardScaler, to_X=True, to_y=True)
dataInverseFlatten = ArrayToDataset(
    shape_X=tmp.X.shape,
    shape_y=tmp.y.shape,
    to_X=True,
    to_y=True,
    axis_names={
        "X": tmp.X.keys(),
        "y": tmp.y.keys()
    },
    flattened=True
)

#%%
tmp_conv = dataFlatten.fit_transform(tmp)
tmp_scaled = scaler.fit_transform(tmp_conv)
tmp_back = dataInverseFlatten.fit_transform(tmp_scaled)
#%%
from caits.transformers._sklearn_wrapper import SklearnWrapper

pipe = SklearnWrapper(
    to_X=False,
    to_y=False,
    sklearn_transformers=[
        ("zscore", StandardScaler, {})
    ]
)
#%%
pipe.fit(tmp)
pipe
#%%
pipe_tmp = pipe.transform(tmp)
pipe_tmp

#%%
tmp.X.iloc[0, :, :]
#%%
pipe_tmp.X.iloc[0, :, :]
#%%
tmp.y.iloc[0, :, :]
#%%
pipe_tmp.y.iloc[0, :, :]
#%% md
# ## ColumnTransformer
#%%
from caits.filtering import filter_median_gen
from caits.transformers import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from caits.properties import magnitude_signal

pipe_filter = Pipeline(
    [
        ("median", FunctionTransformer(filter_median_gen, window_size=20, to_y=True)),
        ("butterworth", FunctionTransformer(filter_butterworth, fs=10, filter_type='highpass', cutoff_freq=2, to_y=True))
    ]
)

pipe_scaler = SklearnPipeStep(StandardScaler)

mag_tr = FeatureExtractorSignal(
    [
        {
            "func": magnitude_signal,
            "params": {
                "axis": 1
            }
        }
    ], axis=1
)

column_tr1 = ColumnTransformer(
    [
        ("filter_acc_x_gyr_x", pipe_filter, ["NO2(GT)", "CO(GT)"], ["new_NO2", "new_CO"]),
        ("filter_acc_y_gyr_y", pipe_filter, ["T", "NMHC(GT)"], ["new_T", "new_NMHC"]),
    ],
    unify=False
)

column_tr2 = ColumnTransformer(
    [
        ("scale_acc_x_acc_y_acc_z", pipe_scaler, ["NO2(GT)", "CO(GT)"], ["scaled_NO2", "scaled_CO"]),
    ],
    unify=True
)

column_tr3 = ColumnTransformer(
    [
        ("mag_calc_1", mag_tr, ["NO2(GT)", "CO(GT)"], ["mag_NO2_CO"]),
        ("mag_calc_2", mag_tr, ["T", "NMHC(GT)"], ["mag_T_NHMC"]),
        ("mag_calc_3", mag_tr, ["scaled_NO2", "scaled_CO"], ["mag_scaled_NO2_CO"]),
    ],
    unify=True
)

final_pipe = Pipeline(
    [
        ("filter", column_tr1),
        ("scale", column_tr2),
        ("mag", column_tr3),
    ]
)

#%%
datasetArrayObj.X
#%%
final_data = final_pipe.fit_transform(datasetArrayObj)
#%%
final_data
#%%
final_data.X
#%%
final_data.y
#%%
final_data.X.shape
#%%
datasetArrayObj.X.values
#%%
final_data.X.values
#%%
final_data.X
#%%
final_data.X.shape, final_data.y.shape
#%% md
# ## Test statistical
#%%
from caits.fe import central_moments
#%%
datasetArrayObj.apply(central_moments)
#%%
central_mom_transformer = FeatureExtractorScalar(
    [
        {
            "func": central_moments
        }
    ],
    to_dataset=True
)

central_mom_transformer.fit_transform(datasetArrayObj)
#%%
from caits.fe import max_value

datasetArrayObj.apply(max_value)
#%%
max_tr = FeatureExtractorScalar(
    [
        {
            "func": max_value
        }
    ],
)

max_tr.fit_transform(datasetArrayObj).X
#%%
from caits.fe import mfcc_mean

datasetArrayObj.apply(mfcc_mean, n_mfcc=5)
#%%
from caits.fe import dominant_frequency

datasetArrayObj.apply(dominant_frequency, fs=50)
#%%
from caits.fe import spectral_kurtosis
datasetArrayObj.apply(spectral_kurtosis, fs=100)
#%%
from caits.fe import (
mean_value,
std_value,
variance_value,
kurtosis_value,
dominant_frequency,
max_value,
average_power,
min_value,
energy,
crest_factor,
sample_skewness,
delta,
envelope_energy_peak_detection,
rms_max,
rms_min,
rms_value,
rms_mean,
zcr_max,
zcr_min,
zcr_value,
zcr_mean,
spectral_bandwidth,
spectral_std,
spectral_values,
spectral_kurtosis,
spectral_slope,
spectral_spread,
spectral_rolloff,
spectral_skewness,
spectral_centroid,
spectral_decrease,
spectral_flatness,
median_value,
signal_length,
max_possible_amplitude,
underlying_spectral
)

scalar_tr = FeatureExtractorScalar(
    [
        {
            "func": mean_value
        },
        {
            "func": std_value
        },
        {
            "func": variance_value
        },
        {
            "func": kurtosis_value
        },
        {
            "func": dominant_frequency,
            "params": {
                "fs": 100
            }
        },
        {
            "func": max_value
        },
        {
            "func": crest_factor
        },
        {
            "func": min_value
        },
        {
            "func": energy
        },
        {
            "func": crest_factor
        },
        {
            "func": average_power
        },
        {
            "func": sample_skewness
        },
        {
            "func": rms_mean,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": rms_value,
        },
        {
            "func": rms_max,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": rms_min,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": zcr_value
        },
        {
            "func": zcr_max,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": zcr_min,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": zcr_mean,
            "params": {
                "frame_length": 20,
                "hop_length": 10
            }
        },
        {
            "func": spectral_bandwidth,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_std,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_kurtosis,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_slope,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_rolloff,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_skewness,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_centroid,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_decrease,
            "params": {
                "fs": 100
            }
        },
        {
            "func": spectral_flatness,
            "params": {
                "fs": 100
            }
        },
        {
            "func": median_value,
        },
        {
            "func": central_moments
        },
        {
            "func": delta,
            "params": {
                "width": 201,
                "order": 1
            }
        }
    ]
)
#%%
tmp = scalar_tr.fit_transform(datasetArrayObj)
tmp.X
#%%
datasetArrayObj.apply(min_value)
#%%
from caits.fe import central_moments

# datasetArrayObj.apply(central_moments)
centrals = central_mom_transformer.fit_transform(datasetArrayObj)
centrals
#%%
centrals.X