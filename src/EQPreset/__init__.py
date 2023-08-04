import struct
import enum
import math

class SupportedEQs(enum.Enum):
    ProQ3 = enum.auto()
    ReaEQ = enum.auto()
    ReaFIR = enum.auto()

class ProQ3Preset:

    class Band:
        class Shape(enum.Enum):
            Bell = 0
            LowShelf = 1
            LowCut = 2
            HighShelf = 3
            HighCut = 4
            Notch = 5
            BandPass = 6
            TiltShelf = 7
            FlatTilt = 8
        
        class Slope(enum.Enum):
            S6dB  = 0
            S12dB = 1
            S18dB = 2
            S24dB = 3
            S30dB = 4
            S36dB = 5
            S48dB = 6
            S72dB = 7
            S96dB = 8
            SBrick = 9
        
        class Placement(enum.Enum):
            Left = 0
            Right = 1
            Stereo = 2
            Mid = 3
            Side = 4
        
        class Speakers(enum.Enum):
            All = 0
            AllExclLFE = 1
            LFE = 2
            Center = 3
            LRFront = 4
            LRFrontCenter = 5
            LRSurroundSide = 6
            LRSurround = 7
            LRSurroundRear = 8
            CenterSurround = 9
            LRTopSurround = 10
        
        class Solo(enum.Enum):
            Disabled = 0
            Enabled = 1
        
        def __init__(self,
                     used: bool = False,
                     enabled: bool = True,
                     frequency: float = 1000.0,
                     gain_db: float = 0.0,
                     dynamic_range_db: float = 0.0,
                     dynamics_enabled: bool = False,
                     dynamic_threshold: float = 0.0,
                     q: float = 1.0,
                     shape: Shape = Shape.Bell,
                     slope: Slope = Slope.S12dB,
                     placement: Placement = Placement.Stereo,
                     speakers: Speakers = Speakers.AllExclLFE,
                     solo: bool = False,
                     external_side_chain: bool = False
                     ) -> None:
            self.used = used
            self.enabled = enabled
            self.frequency = frequency
            self.gain_db = gain_db
            self.dynamic_range_db = dynamic_range_db
            self.dynamics_enabled = dynamics_enabled
            self.dynamic_threshold = dynamic_threshold
            self.q = q
            self.shape = shape
            self.slope = slope
            self.placement = placement
            self.speakers = speakers
            self.solo = solo
            self.external_side_chain = external_side_chain
    
    class Settings:
        class ProcessingMode(enum.Enum):
            ZeroLatency = 0
            NaturalPhase = 1
            LinearPhase = 2
        
        class ProcessingResolution(enum.Enum):
            Low = 0
            Medium = 1
            High = 2
            VeryHigh = 3
            Maximum = 4
        
        class OutputPanMode(enum.Enum):
            LR = 0
            MS = 1
        
        class ExternalSpectrum(enum.Enum):
            Off = -1
            SideChain = 0
        
        class AnalyzerRange(enum.Enum):
            R60dB = 0
            R90dB = 1
            R120dB = 2
        
        class AnalyzerResolution(enum.Enum):
            Low = 0
            Medium = 1
            High = 2
            Maximum = 3
        
        class AnalyzerSpeed(enum.Enum):
            VerySlow = 0
            Slow = 1
            Medium = 2
            Fast = 3
            VeryFast = 4
        
        class AnalyzerTilt(enum.Enum):
            T0dB = 0
            T1dB5 = 1
            T3dB = 2
            T4dB5 = 3
            T6dB = 4
        
        class AnalyzerDisplayRange(enum.Enum):
            R3dB = 0
            R6dB = 1
            R12dB = 2
            R30dB = 3
        
        def __init__(self,
                     processing_mode: ProcessingMode = ProcessingMode.ZeroLatency,
                     processing_resolution: ProcessingResolution = ProcessingResolution.Medium,
                     gain_scale_percent: float = 100.0,
                     output_level_db: float = 0.0,
                     output_pan: float = 0.0,
                     output_pan_mode: OutputPanMode = OutputPanMode.LR,
                     bypass: bool = False,
                     invert_phase: bool = False,
                     auto_gain: bool = False,
                     analyzer_show_pre_processing: bool = True,
                     analyzer_show_post_processing: bool = True,
                     analyzer_external_spectrum: ExternalSpectrum = ExternalSpectrum.Off,
                     analyzer_range: AnalyzerRange = AnalyzerRange.R90dB,
                     analyzer_resolution: AnalyzerResolution = AnalyzerResolution.High,
                     analyzer_speed: AnalyzerSpeed = AnalyzerSpeed.Medium,
                     analyzer_tilt: AnalyzerTilt = AnalyzerTilt.T4dB5,
                     analyzer_freeze: bool = False,
                     analyzer_show_collisions: bool = True,
                     analyzer_spectrum_grab: bool = True,
                     analyzer_display_range: AnalyzerDisplayRange = AnalyzerDisplayRange.R12dB,
                     receive_midi: bool = False,
                     solo_gain_db: float = 0.0
                     ) -> None:
            self.processing_mode = processing_mode
            self.processing_resolution = processing_resolution
            self.gain_scale_percent = gain_scale_percent
            self.output_level_db = output_level_db
            self.output_pan = output_pan
            self.output_pan_mode = output_pan_mode
            self.bypass = bypass
            self.invert_phase = invert_phase
            self.auto_gain = auto_gain
            self.analyzer_show_pre_processing = analyzer_show_pre_processing
            self.analyzer_show_post_processing = analyzer_show_post_processing
            self.analyzer_external_spectrum = analyzer_external_spectrum
            self.analyzer_range = analyzer_range
            self.analyzer_resolution = analyzer_resolution
            self.analyzer_speed = analyzer_speed
            self.analyzer_tilt = analyzer_tilt
            self.analyzer_freeze = analyzer_freeze
            self.analyzer_show_collisions = analyzer_show_collisions
            self.analyzer_spectrum_grab = analyzer_spectrum_grab
            self.analyzer_display_range = analyzer_display_range
            self.receive_midi = receive_midi
            self.solo_gain_db = solo_gain_db

    def __init__(self) -> None:
        self.header = "FQ3p"
        self.version_major = 4
        self.version_minor = 358
        
        self.bands = [ProQ3Preset.Band() for i in range(24)]
        self.settings = ProQ3Preset.Settings()
    
    def save_to_file(self, filename: str):
        data = struct.pack(
            "<4sii",
            bytes(self.header, "ASCII"),
            self.version_major,
            self.version_minor
        )
        for band in self.bands:
            data += struct.pack(
                "<fffffffffffff",
                float(band.used),
                float(band.enabled),
                self.FreqConvert(band.frequency),
                band.gain_db,
                band.dynamic_range_db,
                float(band.dynamics_enabled),
                self.ThresholdConvert(band.dynamic_threshold),
                self.QConvert(band.q),
                float(band.shape.value),
                float(band.slope.value),
                float(band.placement.value),
                float(band.speakers.value),
                float(band.solo)
            )
        data += struct.pack(
            "<ffffffffffffffffffffff",
            float(self.settings.processing_mode.value),
            float(self.settings.processing_resolution.value),
            self.GainScaleConvert(self.settings.gain_scale_percent),
            self.settings.output_level_db,
            self.settings.output_pan,
            float(self.settings.output_pan_mode.value),
            float(self.settings.bypass),
            float(self.settings.invert_phase),
            float(self.settings.auto_gain),
            float(self.settings.analyzer_show_pre_processing),
            float(self.settings.analyzer_show_post_processing),
            float(self.settings.analyzer_external_spectrum.value),
            float(self.settings.analyzer_range.value),
            float(self.settings.analyzer_resolution.value),
            float(self.settings.analyzer_speed.value),
            float(self.settings.analyzer_tilt.value),
            float(self.settings.analyzer_freeze),
            float(self.settings.analyzer_show_collisions),
            float(self.settings.analyzer_spectrum_grab),
            float(self.settings.analyzer_display_range.value),
            float(self.settings.receive_midi),
            self.settings.solo_gain_db
        )
        for band in self.bands:
            data += struct.pack("<f", band.external_side_chain)
        with open(f"{filename}.ffp", "wb") as f:
            f.write(data)

    @staticmethod
    def FreqConvert(frequency: float) -> float:
        return math.log10(frequency) / math.log10(2)

    @staticmethod
    def QConvert(q: float) -> float:
        q = 0.025 if q < 0.025 else (40.0 if q > 40.0 else q)
        return (math.log10(q) * (1 / math.log10(40.0 / 0.025))) + 0.5
    
    @staticmethod
    def ThresholdConvert(threshold: float) -> float:
        if threshold >= -1.0:
            return 1.0
        elif threshold >= -48:
            return 1.0 - ((0.8 / 48.0) * abs(threshold))
        elif threshold >= -72:
            return 0.2 - ((0.1 / 24.0) * (abs(threshold) - 48.0))
        elif threshold > 90:
            return 0.1 - ((0.1 / 18.0) * (abs(threshold) - 72.0))
        else:
            return 0.0
    
    @staticmethod
    def GainScaleConvert(gain_scale: float) -> float:
        return gain_scale / 100.0

class FXP1Preset:
    def __init__(self, preset_name: str, fx_id: str, fx_version: int, ) -> None:
        self.chunk_magic = "CcnK"
        self.byte_size = -1
        self.fx_magic = "FPCh"
        self.format_version = 1
        self.fx_id = fx_id
        self.fx_version = fx_version
        self.program_count = 1
        self.name = preset_name
        self.chunk_size = -1
    
    def save_to_file(self, filename: str, chunk_data: bytes):
        chunk_size = len(chunk_data)
        header = struct.pack(
            ">4si4sii28si",
            bytes(self.fx_magic, "ASCII"),
            self.format_version,
            bytes(self.fx_id, "ASCII"),
            self.fx_version,
            self.program_count,
            bytes(self.name, "ASCII"),
            chunk_size
        )
        header_size = len(header)
        header = struct.pack(
            ">4si",
            bytes(self.chunk_magic, "ASCII"),
            header_size
        ) + header + chunk_data
        with open(f"{filename}.fxp", "wb") as f:
            f.write(header)
    
    @staticmethod
    def GainTodB(gain: float) -> float:
        return 20 * math.log10(gain)

    @staticmethod
    def dBToGain(dB: float) -> float:
        return math.pow(10, dB / 20)
    
    @staticmethod
    def qToBandwidth(q: float) -> float:
        y = (2*q**2+1)/(2*q**2)+math.sqrt(((((2*q**2+1)/q**2)**2)/4)-1)
        numerator = math.log10(y)
        denominator = math.log10(2)
        return numerator / denominator
    
    @staticmethod
    def BandwithToQ(bandwidth: float) -> float:
        numerator = math.sqrt(2 ** bandwidth)
        denominator = (2 ** bandwidth) - 1
        return numerator / denominator

class ReaEQPreset(FXP1Preset):
    
    class Band:

        class Shape(enum.Enum):
            LowShelf = 0
            HighShelf = 1
            Band_old = 2
            LowPass = 3
            HighPass = 4
            AllPass = 5
            Notch = 6
            BandPass = 7
            Band = 8
            Band_alt = 9

        def __init__(self, 
                     shape: Shape = Shape.Band,
                     enabled: bool = True,
                     frequency: float = 1000,
                     gain_db: float = 0.0,
                     q: float = 2.0,
                     log_scale_automated: bool = True
                     ) -> None:
            self.shape = shape
            self.enabled = enabled
            self.frequency = frequency
            self.gain_db = gain_db
            self.q = q
            self.log_scale_automated = log_scale_automated

    class Settings:
        class Metering(enum.Enum):
            All = 0
            Non = 1
            Ch1 = 2
            Ch2 = 3
            Ch3 = 4
            Ch4 = 5
            Ch5 = 6
            Ch6 = 7
            Ch7 = 8
            Ch8 = 9
            Ch9 = 10
        
        def __init__(self,
                     show_grid: bool = True,
                     show_tabs: bool = True,
                     output_gain_db: float = 0.0,
                     show_phase: bool = False,
                     width: int = 576,
                     height: int = 363,
                     metering: Metering = Metering.All
                     ) -> None:
            self.show_grid = show_grid
            self.show_tabs = show_tabs
            self.output_gain_db = output_gain_db
            self.show_phase = show_phase
            self.width = width
            self.height = height
            self.metering = metering

    def __init__(self, 
                 preset_name: str,
                 show_grid: bool = True,
                 show_tabs: bool = True,
                 output_gain_db: float = 0.0,
                 show_phase: bool = False,
                 width: int = 576,
                 height: int = 363,
                 metering: Settings.Metering = Settings.Metering.All
                 ) -> None:
        super().__init__(preset_name, "reeq", 1100)        
        self.bands = []
        self.settings = ReaEQPreset.Settings(show_grid,
                                             show_tabs,
                                             output_gain_db,
                                             show_phase,
                                             width,
                                             height,
                                             metering)
    
    def add_band(self, shape: Band.Shape, enabled: bool, frequency: float, gain_db: float, q: float, log_scale_automated: bool = True) -> int:
        index = len(self.bands)
        self.bands.append(
            ReaEQPreset.Band(
                shape,
                enabled,
                frequency,
                gain_db,
                q,
                log_scale_automated
            )
        )
        return index
    
    def remove_band(self, index):
        self.bands.pop(index)
    
    def save_to_file(self, filename: str):
        band_count = len(self.bands)
        band_size = 33
        self.chunk_data = struct.pack(
            "<ii",
            band_size,
            band_count
        )
        band: ReaEQPreset.Band
        for band in self.bands:
            self.chunk_data += struct.pack(
                "<iidddb", 
                int(band.shape.value),
                int(band.enabled),
                band.frequency,
                self.dBToGain(band.gain_db),
                self.qToBandwidth(band.q),
                int(band.log_scale_automated)
            )
        self.chunk_data += struct.pack(
            "<iidiiibbbb", 
            int(self.settings.show_grid),
            int(self.settings.show_tabs),
            self.dBToGain(self.settings.output_gain_db),
            int(self.settings.show_phase),
            self.settings.width,
            self.settings.height,
            2,
            int(self.settings.metering.value),
            0,
            0
        )
        super().save_to_file(filename, self.chunk_data)

class ReaFIRPreset(FXP1Preset):

    class Settings:
        class ChunkFormat(enum.Enum):
            Points = 4764
            Precise = 4763
        
        class EditMode(enum.Enum):
            Flat = 0
            Precise = 1
            Smooth = 2

        class Quality(enum.Enum):
            Best = 0.125
            Legacy0716 = 0
            Legacy0607 = 1
        
        class Mode(enum.Enum):
            EQ = 0
            Gate = 1
            Compressor = 2
            ConvolveLR = 3
            Subtract = 4
        
        class Metering(enum.Enum):
            Average = 0
            Sum = 1
            Max = 2
            Non = 3
            Ch1 = 4
            Ch2 = 5
            Ch12 = 6
        
        def __init__(self,
                     display_bottom: float = -90,
                     display_top: float = 24.0,
                     fft_size: int = 4096,
                     show_analysis_floor: bool = True,
                     reduce_artifacts: bool = False,
                     edit_mode: EditMode = EditMode.Smooth,
                     mode: Mode = Mode.EQ,
                     gate_floor_db: float = -90,
                     compressor_ratio: float = 1.0,
                     output_gain_db: float = 0,
                     analysis_floor_db: float = -90,
                     quality: Quality = Quality.Best,
                     metering: Metering = Metering.Average
                    ) -> None:
            self.display_bottom = display_bottom
            self.display_top = display_top
            self.fft_size = fft_size
            self.show_analysis_floor = show_analysis_floor
            self.reduce_artifacts = reduce_artifacts
            self.edit_mode = edit_mode
            self.mode = mode
            self.gate_floor_db = gate_floor_db
            self.compressor_ratio = compressor_ratio
            self.output_gain_db = output_gain_db
            self.analysis_floor_db = analysis_floor_db
            self.quality = quality
            self.metering = metering

    class Band:
        def __init__(self,
                     frequency: float = 1000.0,
                     gain_db: float = 0.0) -> None:
            self.frequency = frequency
            self.gain_db = gain_db

    def __init__(self, 
                 preset_name: str,
                 display_bottom: float = -90,
                 display_top: float = 24.0,
                 fft_size: int = 4096,
                 show_analysis_floor: bool = True,
                 reduce_artifacts: bool = False,
                 edit_mode: Settings.EditMode = Settings.EditMode.Smooth,
                 mode: Settings.Mode = Settings.Mode.EQ,
                 gate_floor_db: float = -90,
                 compressor_ratio: float = 1.0,
                 output_gain_db: float = 0,
                 analysis_floor_db: float = -90,
                 quality: Settings.Quality = Settings.Quality.Best,
                 metering: Settings.Metering = Settings.Metering.Average,
                 ) -> None:
        super().__init__(preset_name, "refr", 1100)
        self.settings = ReaFIRPreset.Settings(display_bottom,
                                              display_top,
                                              fft_size,
                                              show_analysis_floor,
                                              reduce_artifacts,
                                              edit_mode,
                                              mode,
                                              gate_floor_db,
                                              compressor_ratio,
                                              output_gain_db,
                                              analysis_floor_db,
                                              quality,
                                              metering)
        self.bands = []
    
    def add_band(self, frequency: float, gain_db: float):
        index = len(self.bands)
        self.bands.append(ReaFIRPreset.Band(frequency, gain_db))
        return index

    def remove_band(self, index: int):
        self.bands.pop(index)

    def save_to_file(self, filename: str):
        chunk_data = struct.pack(
            "<iffiffii",
            ReaFIRPreset.Settings.ChunkFormat.Points.value,
            self.settings.display_bottom,
            self.settings.display_top,
            self.settings.fft_size,
            float(self.settings.show_analysis_floor),
            float(self.settings.reduce_artifacts),
            self.settings.edit_mode.value,
            len(self.bands)
        )
        band: ReaFIRPreset.Band
        for band in self.bands:
            chunk_data += struct.pack(
                "<ff",
                band.frequency,
                band.gain_db
            )
        chunk_data += struct.pack(
            "<ifffffiihhf",
            self.settings.mode.value,
            self.dBToGain(self.settings.gate_floor_db),
            self.RatioToOne(self.settings.compressor_ratio),
            self.dBToGain(self.settings.output_gain_db),
            self.dBToGain(self.settings.analysis_floor_db),
            self.settings.quality.value,
            0,
            0,
            2,
            self.settings.metering.value,
            1.0
        )
        super().save_to_file(filename, chunk_data)
    
    @staticmethod
    def RatioToOne(ratio: float) -> float:
        return (ratio - 0.2) / 99.8