import wave as wav
import numpy as np
import scipy.io.wavfile as sciowav



""" Public Methods """

class WavIO:
# Read a wav file, and output an appropriate amount of wav files

    def __init__(self, fn):

        self.fn = fn
        self.in_file = wav.open(fn, mode='r')

    def read_source(self):
    # Returns an array of numpy vectors representing a source in
    # the .wav file
    # Vector elements are ?-bit signed integers

        # Open file and read it
        frame_size = self.in_file.getnframes()
        channel_size = self.in_file.getnchannels()
        if channel_size == 1: raise BaseException('Must be more than 1 channel')
        self.rate, data = sciowav.read(self.fn) 

        # Check if compressed
        _compressed_error_check(self.in_file)

        # Remember data type
        self.dtype = data.dtype

        # Seperate each channel source in a seperate array
        return np.array(data, dtype=np.float64)

    def write_sources(self, sources):
    # Writes a wav file named "fn_#.wav" for all N sources

        root_name = self.fn[:-4]+'_'
        for i in range(len(sources)):
            source = _normalize_volume(sources[i])
            # Catch special case of uint8
            if source.dtype == self.dtype:
                source = np.array(source, dtype=self.dtype)
            sciowav.write(root_name+str(i)+'.wav', self.rate, source)
            
def scale_source(source, factor):
# Scales a source by a factor of factor and returns the result

    return np.array(source*factor, dtype=source.dtype)

""" Private Methods """

def _normalize_volume(source):
# Increases or decreases volume back to an appropriate scale

    # Find the maximum value
    high = max(source)
    low = min(source)
    max_val = max(abs(high), abs(low))

    # docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
    type_name = str(source.dtype) 
    peak = 0
    if   type_name == 'uint8':      peak = 255
    elif type_name == 'int16':      peak = 32767
    elif type_name == 'int32':      peak = 2147483647
    elif type_name == 'float32':    peak = 1

    # Return the scaled source so that the max value touches
    # the max value allowed by the wav file type
    return scale_source(source, peak/max_val)


def _compressed_error_check(in_file):
# Raises an Exception on compressed files

    comp_type = in_file.getcompname()
    if comp_type != 'None' and comp_type != 'not compressed':
        raise BaseException('Compression Type is: ' + comp_type)


class _sign_extend():
# DON'T WORRY ABOUT THIS BRAD.  DIDN'T USE IT
# Given a byte, convert to a signed integer

    def __init__(self, sample_width):

        self.max_val = 2 ** sample_width
        self.max_pos = self.max_val/2 - 1

    def __call__(self, data):

        # Cast
        integer = int(data)
        # Negative case
        if integer > self.max_pos:
            return integer - self.max_val
        # Positive case
        return integer

if __name__ == '__main__':

    io = WavIO('elvis_riverside.wav')
    channels = io.read_source()
    channels = [ channels[0][:int(len(channels[0])/2)] ]
    io.write_sources(channels)
