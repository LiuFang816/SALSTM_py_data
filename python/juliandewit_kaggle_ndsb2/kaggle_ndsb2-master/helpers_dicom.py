__author__ = 'Julian'
import dicom
import numpy


class DicomWrapper:
    def __init__(self, file_dir, file_name):
        self.raw_file = dicom.read_file(file_dir + file_name)
        self.file_name = file_name
        loc1 = self.get_location()
        loc2 = self.slice_location
        # print str(loc2) + " - " + str(loc1)

    def get_value(self, name):
        res = self.raw_file.data_element(name).value
        return res

    @property
    def columns(self):
        res = self.get_value("Columns")
        return res

    @property
    def rows(self):
        res = self.get_value("Rows")
        return res

    @property
    def spacing(self):
        res = self.get_value("PixelSpacing")
        return res

    @property
    def slice_location(self):
        return self.get_value("SliceLocation")

    @property
    def create_time(self):
        return str(int(round(float(self.get_value("InstanceCreationTime")))) / 10).rjust(5, '0')

    @property
    def slice_thickness(self):
        return self.get_value("SliceThickness")

    @property
    def sequence_name(self):
        return self.get_value("SequenceName")

    @property
    def image_position(self):
        return self.get_value("ImagePositionPatient")

    @property
    def series_number(self):
        return self.get_value("SeriesNumber")

    @property
    def series_time(self):
        return self.get_value("SeriesTime")

    @property
    def patient_id(self):
        return self.get_value("PatientID")

    @property
    def series_description(self):
        return self.get_value("SeriesDescription")

    @property
    def image_orientation_patient(self):
        return self.get_value("ImageOrientationPatient")

    @property
    def image_position_patient(self):
        return self.get_value("ImagePositionPatient")

    @property
    def flip_angle(self):
        return self.get_value("FlipAngle")

    @property
    def instance_number(self):
        return self.get_value("InstanceNumber")

    @property
    def in_plane_encoding_direction(self):
        return self.get_value("InPlanePhaseEncodingDirection")

    def get_location(self):
        image_center2d = self.spacing * (numpy.array([self.columns, self.rows]) - numpy.ones(2)) / 2.
        image_center3d = numpy.dot(image_center2d, numpy.reshape(self.image_orientation_patient, (2, 3)))
        center = self.image_position_patient + image_center3d
        direction = numpy.argmax(numpy.abs(numpy.cross(self.image_orientation_patient[:3], self.image_orientation_patient[3:])))
        res = numpy.round(center[direction], 2)
        return center

    @property
    def pixel_array(self):
        img = self.raw_file.pixel_array.astype(float) / numpy.max(self.raw_file.pixel_array)
        return img

    def get_csv(self):
        res = [self.series_number, self.get_value("InstanceNumber"), self.flip_angle, self.series_description, self.series_time]
        return res

    def dir(self):
        self.raw_file.dir()





#
# ['AcquisitionMatrix',
#  'AcquisitionNumber',
#  'AcquisitionTime',
#  'AngioFlag',
#  'BitsAllocated',
#  'BitsStored',
#  'BodyPartExamined',
#  'CardiacNumberOfImages',
#  'Columns',
#  'CommentsOnThePerformedProcedureStep',
#  'EchoNumbers',
#  'EchoTime',
#  'EchoTrainLength',
#  'FlipAngle',
#  'HighBit',
#  'ImageOrientationPatient',
#  'ImagePositionPatient',
#  'ImageType',
#  'ImagedNucleus',
#  'ImagingFrequency',
#  'InPlanePhaseEncodingDirection',
#  'InstanceCreationTime',
#  'InstanceNumber',
#  'LargestImagePixelValue',
#  'MRAcquisitionType',
#  'MagneticFieldStrength',
#  'Manufacturer',
#  'ManufacturerModelName',
#  'Modality',
#  'NominalInterval',
#  'NumberOfAverages',
#  'NumberOfPhaseEncodingSteps',
#  'PatientAddress',
#  'PatientAge',
#  'PatientBirthDate',
#  'PatientID',
#  'PatientName',
#  'PatientPosition',
#  'PatientSex',
#  'PatientTelephoneNumbers',
#  'PercentPhaseFieldOfView',
#  'PercentSampling',
#  'PerformedProcedureStepID',
#  'PerformedProcedureStepStartTime',
#  'PhotometricInterpretation',
#  'PixelBandwidth',
#  'PixelData',
#  'PixelRepresentation',
#  'PixelSpacing',
#  'PositionReferenceIndicator',
#  'RefdImageSequence',
#  'ReferencedImageSequence',
#  'RepetitionTime',
#  'Rows',
#  'SAR',
#  'SOPClassUID',
#  'SOPInstanceUID',
#  'SamplesPerPixel',
#  'ScanOptions',
#  'ScanningSequence',
#  'SequenceName',
#  'SequenceVariant',
#  'SeriesDescription',
#  'SeriesNumber',
#  'SeriesTime',
#  'SliceLocation',
#  'SliceThickness',
#  'SmallestImagePixelValue',
#  'SoftwareVersions',
#  'SpecificCharacterSet',
#  'StudyTime',
#  'TransmitCoilName',
#  'TriggerTime',
#  'VariableFlipAngleFlag',
#  'WindowCenter',
#  'WindowCenterWidthExplanation',
#  'WindowWidth',
#  'dBdt']