?$	??z?;^@?Β.`?p@???h?x??!?s)?*܂@	!       "j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)?s)?*܂@U?]}@1a?9??@I!>???@??r1"j
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)??Bt?????b?DS?18????C??Is/0+???r2"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???h?x??1!??????I?YL???r3"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?wak????1??A???I?Ac&Q??r4"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails cD?в??1sd?????I?	K<?l??r5*	*?َ$?@2?
OIterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2 ??ao
@!EU?=??T@)??ao
@1EU?=??T@:Preprocessing2t
=Iterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle::Zip ?Fˁ?@!???z?xV@)=,Ԛ???1??JF@:Preprocessing2f
/Iterator::Root::FiniteRepeat::Prefetch::BatchV2??c"?@!????W@)W@????1^?????@:Preprocessing2?
\Iterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice ?b.???!???G	@)?b.???1???G	@:Preprocessing2?
MIterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSlice ????V%??!?mp]{?@)????V%??1?mp]{?@:Preprocessing2o
8Iterator::Root::FiniteRepeat::Prefetch::BatchV2::Shuffle <?ן?g@!??w?OW@)+P??ô??1??ǽ??@:Preprocessing2S
Iterator::Root::FiniteRepeataS?Q???!????!l??)w??-u???1S??[y??:Preprocessing2]
&Iterator::Root::FiniteRepeat::PrefetchK %vmo??!?A^?/_??)K %vmo??1?A^?/_??:Preprocessing2E
Iterator::RootձJ??^??!q&'o???)???D?v?1@??k???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Ow̤???Q?"?lٹX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)??0???YN?T?_??!U?]}@	!       "$	x??4?]@h??ʵp@!??????!a?9??@*	!       2	!       :$	_;?????)vE????YL???!!>???@??B	!       J	!       R	!       Z	!       b	!       JGPUb q?Ow̤???y?"?lٹX@?	"t
Xgradient_tape/model/bert/encoder/layer_._10/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul\?n?}?!\?n?}?"s
Wgradient_tape/model/bert/encoder/layer_._7/intermediate/dense/Tensordot/MatMul/MatMul_1MatMulb??rͦ}?!_ifp竍?"t
Xgradient_tape/model/bert/encoder/layer_._11/intermediate/dense/Tensordot/MatMul/MatMul_1MatMulnUE	?}?!?v	?>??"s
Wgradient_tape/model/bert/encoder/layer_._5/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul?S?}?!?8?????"s
Wgradient_tape/model/bert/encoder/layer_._0/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul8$J???}?!
???/???"s
Wgradient_tape/model/bert/encoder/layer_._2/intermediate/dense/Tensordot/MatMul/MatMul_1MatMulRX`?k?}?!?\=9??"s
Wgradient_tape/model/bert/encoder/layer_._3/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul????x}?!?sZV???"s
Wgradient_tape/model/bert/encoder/layer_._1/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul?55Ȼp}?!G?m???"m
Qgradient_tape/model/bert/encoder/layer_._5/output/dense/Tensordot/MatMul/MatMul_1MatMul?'??a}?!4?,T???"s
Wgradient_tape/model/bert/encoder/layer_._9/intermediate/dense/Tensordot/MatMul/MatMul_1MatMul-)??;*x?!?X??#??Q      Y@YhRh}?@a~?z)??W@q{??=??@y?^?? Z7?"?	
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 