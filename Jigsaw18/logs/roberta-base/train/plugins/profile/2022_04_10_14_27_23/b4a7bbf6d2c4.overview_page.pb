?$	m???1^@1Dko?p@??հߣ?!?0?1ق@	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?i4???1?7?0???I??????r1"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??հߣ?1??+ٱq?I????u???r2"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?ȭI?%??1mU?Y??IN~?N?Z??r3"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)?0?1ق@?z?<d?@1????ǝ?@I?jH?c???r4"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??p???1X?|[?Tw?I?h8en???r5*	-????,p@2f
/Iterator::Root::FiniteRepeat::Prefetch::BatchV2?Kp????!9?m?uS@)YL?Q???1(?P?M@:Preprocessing2o
8Iterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle ?Ր??ҧ?!??????1@)?Ր??ҧ?1??????1@:Preprocessing2S
Iterator::Root::FiniteRepeatL?
F%u??!?????3@)]T????1?r˲H'@:Preprocessing2]
&Iterator::Root::FiniteRepeat::Prefetch<3?p?a??!?(gA? @)<3?p?a??1?(gA? @:Preprocessing2E
Iterator::Root?_>Y1\??!'?K?(6@)X˝?`8w?1?R??I?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI E??|??Q???A?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???0P???X??1)@!?z?<d?@	!       "$	<?$?]@U????p@??+ٱq?!????ǝ?@*	!       2	!       :$	N??ĭ????l?????????u???!?jH?c???B	!       J	!       R	!       Z	!       b	!       JGPUb q E??|??y???A?X@?	"v
Zgradient_tape/model/roberta/encoder/layer_._1/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul??I???}?!??I???}?"v
Zgradient_tape/model/roberta/encoder/layer_._3/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul??wZ?}?!L?????"v
Zgradient_tape/model/roberta/encoder/layer_._0/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul???D??}?!9R?P??"q
Ugradient_tape/model/roberta/encoder/layer_._10/output/dense/Tensordot/MatMul/MatMul_1MatMul??UpƲ}?!J?/?Y???"v
Zgradient_tape/model/roberta/encoder/layer_._6/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul??ׄϑ}?!#Ͳ?搢?"v
Zgradient_tape/model/roberta/encoder/layer_._9/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul[??*=?}?!????B??"v
Zgradient_tape/model/roberta/encoder/layer_._2/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul?\?h?}?!?<??[???"v
Zgradient_tape/model/roberta/encoder/layer_._4/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul(??F?9x?!??/????"v
Zgradient_tape/model/roberta/encoder/layer_._7/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul?i??6x?!??(0??"w
[gradient_tape/model/roberta/encoder/layer_._11/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul???24x?!Ձ?Is???Q      Y@Y??(m?j??a?]K*U?X@q"?A?J?;@y?{q???7?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 