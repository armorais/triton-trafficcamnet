import sys
import numpy as np
import tritongrpcclient
import tritonshmutils.shared_memory as shm
import tritonclientutils as utils
import operator
from functools import reduce
from utils.detectnetv2_postprocess import DetectNetV2PostProcess, NMS

class Detectnetv2GrpcClient():
    def __init__(self, triton_cfg):
        self.triton_cfg = triton_cfg
        self.batch_size = triton_cfg['model']['batch_size']
        self.model_name = triton_cfg['model']['name']
        self.model_version = triton_cfg['model']['version']
        self.conf_threshold = triton_cfg['model']['conf_threshold']
        self.iou_threshold = triton_cfg['model']['iou_threshold']
        self.input_handles = {}
        self.output_handles = {}
        self.input_layers = []
        self.output_layers = []
        self.postprocessor = DetectNetV2PostProcess(width=960, height=544) # TODO : fix hardcoded dims
        url=triton_cfg['server']['host']+':'+triton_cfg['server']['port']
        try:
            self.triton_client = tritongrpcclient.InferenceServerClient(
                    url=url,
                    verbose=triton_cfg['server']['verbose'])
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
        self.model_config = self.triton_client.get_model_config(triton_cfg['model']['name'],
                                                                triton_cfg['model']['version'],
                                                                as_json=True)
        self._unregister_shm_regions()
        self.input_shape = []
        self._initialize_model()

    # python > 3.8 has a math.prod() function!
    def _prod(self, iterable):
        return reduce(operator.mul, iterable, 1)

    def _initialize_model(self):
        input_cfg = self.model_config['config']['input']
        output_cfg = self.model_config['config']['output']

        input_names = [i['name'] for i in input_cfg]
        output_names = [o['name'] for o in output_cfg]
        print('Input layers: ', output_names)
        print('Output layers: ', output_names)

        input_dims = [[int(dim) for dim in input_cfg[i]['dims']] for i in range(len(input_cfg))]
        output_dims = [[int(dim) for dim in output_cfg[i]['dims']] for i in range(len(output_cfg))]
        self.input_shape = input_dims[0]
        self.output_dims = output_dims

        if self.triton_cfg['model']['precision'] == "FP32":
            mult = 4
        elif self.triton_cfg['model']['precision'] == "FP16":
            mult = 2 # TODO: Fix this
        elif self.triton_cfg['model']['precision'] == "INT8":
            mult = 1 # TODO: Fix this
        else:
            print("unsupported precision in config file: " +
                    str(self.triton_cfg['model']['precision']))
            sys.exit()

        input_byte_sizes_list = [self._prod(dims) * mult for dims in input_dims]
        output_byte_sizes_list = [self._prod(dims) * mult for dims in output_dims]

        for i in range(len(input_cfg)):
            shm_region_name = self.model_name + "_input" + str(i)
            self._register_system_shm_regions(shm_region_name, self.input_handles,
                                              input_byte_sizes_list[i], input_names[i])
            self.input_layers.append(tritongrpcclient.InferInput(input_names[i],
                                    [1,input_dims[i][0],input_dims[i][1],input_dims[i][2]], "FP32"))
            self.input_layers[-1].set_shared_memory(shm_region_name, input_byte_sizes_list[i])

        for i in range(len(output_cfg)):
            shm_region_name = self.model_name + "_output" + str(i)
            self._register_system_shm_regions(shm_region_name, self.output_handles,
                                              output_byte_sizes_list[i], output_names[i])
            self.output_layers.append(tritongrpcclient.InferRequestedOutput(output_names[i]))
            self.output_layers[-1].set_shared_memory(shm_region_name, output_byte_sizes_list[i])

    def _unregister_shm_regions(self):
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

    def _register_system_shm_regions(self, shm_region_name, handles, 
                                    region_byte_size, layer_io_name):
        cuda_shm_handle = shm.create_shared_memory_region(shm_region_name,
                                                        "/" + shm_region_name,
                                                        region_byte_size)
        handles[layer_io_name] = cuda_shm_handle
        self.triton_client.register_system_shared_memory(shm_region_name,
                                                "/" + shm_region_name,
                                                region_byte_size)

    def infer_dims(self):
        return (self.input_shape[2],self.input_shape[1])
    
    def _preprocess_image(self, raw_img):
        image = raw_img[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = np.true_divide(image, 255.0) 
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype=np.float32, order='C')
        return image
        
    # inference
    def __call__(self, raw_img):
        input_image = self._preprocess_image(raw_img)
        shm.set_shared_memory_region(self.input_handles["input_1"], [input_image])

        outputs = self.triton_client.infer(model_name=self.model_name, model_version=self.model_version,
                                            inputs=self.input_layers, outputs=self.output_layers)
        coverages_output = outputs.get_output("output_cov/Sigmoid")
        bboxes_output = outputs.get_output("output_bbox/BiasAdd")

        if coverages_output is not None:
            coverages = shm. get_contents_as_numpy(
                self.output_handles["output_cov/Sigmoid"], utils.triton_to_np_dtype(coverages_output.datatype),
                self._prod(coverages_output.shape))
        else:
            raise Exception("output_cov/Sigmoid layer data is missing in the response.") 

        if bboxes_output is not None:
            bboxes = shm.get_contents_as_numpy(
                self.output_handles["output_bbox/BiasAdd"], utils.triton_to_np_dtype(bboxes_output.datatype),
                self._prod(bboxes_output.shape))
        else:
            raise Exception("output_bbox/BiasAdd layer data is missing in the response.") 

        boxes = self.postprocessor.start(bboxes, coverages)
        boxes = NMS.filter(boxes)
        return boxes
