╞Ы
Ю Б 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
q
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
╖
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
и
SparseSegmentMean	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.02unknown8ам
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:

*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:
*
dtype0
╥
:Adam/dense_features_1/userId_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▒ъ
*K
shared_name<:Adam/dense_features_1/userId_embedding/embedding_weights/v
╦
NAdam/dense_features_1/userId_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOp:Adam/dense_features_1/userId_embedding/embedding_weights/v* 
_output_shapes
:
▒ъ
*
dtype0
╧
9Adam/dense_features/movieId_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	щ
*J
shared_name;9Adam/dense_features/movieId_embedding/embedding_weights/v
╚
MAdam/dense_features/movieId_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOp9Adam/dense_features/movieId_embedding/embedding_weights/v*
_output_shapes
:	щ
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:

*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:
*
dtype0
╥
:Adam/dense_features_1/userId_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▒ъ
*K
shared_name<:Adam/dense_features_1/userId_embedding/embedding_weights/m
╦
NAdam/dense_features_1/userId_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOp:Adam/dense_features_1/userId_embedding/embedding_weights/m* 
_output_shapes
:
▒ъ
*
dtype0
╧
9Adam/dense_features/movieId_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	щ
*J
shared_name;9Adam/dense_features/movieId_embedding/embedding_weights/m
╚
MAdam/dense_features/movieId_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOp9Adam/dense_features/movieId_embedding/embedding_weights/m*
_output_shapes
:	щ
*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:╚*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:╚*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:╚*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:╚*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:╚*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:╚*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:╚*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:

*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
─
3dense_features_1/userId_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▒ъ
*D
shared_name53dense_features_1/userId_embedding/embedding_weights
╜
Gdense_features_1/userId_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp3dense_features_1/userId_embedding/embedding_weights* 
_output_shapes
:
▒ъ
*
dtype0
┴
2dense_features/movieId_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	щ
*C
shared_name42dense_features/movieId_embedding/embedding_weights
║
Fdense_features/movieId_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp2dense_features/movieId_embedding/embedding_weights*
_output_shapes
:	щ
*
dtype0
r
serving_default_movieIdPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
q
serving_default_userIdPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
Ъ
StatefulPartitionedCallStatefulPartitionedCallserving_default_movieIdserving_default_userId2dense_features/movieId_embedding/embedding_weights3dense_features_1/userId_embedding/embedding_weightsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_302825

NoOpNoOp
ЇP
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*пP
valueеPBвP BЫP
╢
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
▀
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources
'#movieId_embedding/embedding_weights*
▐
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_feature_columns
"
_resources
&#"userId_embedding/embedding_weights*
О
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
ж
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias*
ж
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
ж
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
<
0
#1
02
13
84
95
@6
A7*
<
0
#1
02
13
84
95
@6
A7*
* 
░
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
* 
ф
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemЬ#mЭ0mЮ1mЯ8mа9mб@mвAmгvд#vе0vж1vз8vи9vй@vкAvл*

Tserving_default* 

0*

0*
* 
У
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ztrace_0
[trace_1* 

\trace_0
]trace_1* 
* 
* 
бЪ
VARIABLE_VALUE2dense_features/movieId_embedding/embedding_weightsTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*

#0*

#0*
* 
У
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

ctrace_0
dtrace_1* 

etrace_0
ftrace_1* 
* 
* 
бЪ
VARIABLE_VALUE3dense_features_1/userId_embedding/embedding_weightsSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

ltrace_0* 

mtrace_0* 

00
11*

00
11*
* 
У
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
У
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

ztrace_0* 

{trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
Ф
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

Бtrace_0* 

Вtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*
$
Г0
Д1
Е2
Ж3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
З	variables
И	keras_api

Йtotal

Кcount*
M
Л	variables
М	keras_api

Нtotal

Оcount
П
_fn_kwargs*
z
Р	variables
С	keras_api
Тtrue_positives
Уtrue_negatives
Фfalse_positives
Хfalse_negatives*
z
Ц	variables
Ч	keras_api
Шtrue_positives
Щtrue_negatives
Ъfalse_positives
Ыfalse_negatives*

Й0
К1*

З	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Н0
О1*

Л	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Т0
У1
Ф2
Х3*

Р	variables*
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
Ш0
Щ1
Ъ2
Ы3*

Ц	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
─╜
VARIABLE_VALUE9Adam/dense_features/movieId_embedding/embedding_weights/mplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
─╜
VARIABLE_VALUE:Adam/dense_features_1/userId_embedding/embedding_weights/molayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
─╜
VARIABLE_VALUE9Adam/dense_features/movieId_embedding/embedding_weights/vplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
─╜
VARIABLE_VALUE:Adam/dense_features_1/userId_embedding/embedding_weights/volayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFdense_features/movieId_embedding/embedding_weights/Read/ReadVariableOpGdense_features_1/userId_embedding/embedding_weights/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOpMAdam/dense_features/movieId_embedding/embedding_weights/m/Read/ReadVariableOpNAdam/dense_features_1/userId_embedding/embedding_weights/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOpMAdam/dense_features/movieId_embedding/embedding_weights/v/Read/ReadVariableOpNAdam/dense_features_1/userId_embedding/embedding_weights/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_303837
▌	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename2dense_features/movieId_embedding/embedding_weights3dense_features_1/userId_embedding/embedding_weightsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcounttrue_positives_1true_negatives_1false_positives_1false_negatives_1true_positivestrue_negativesfalse_positivesfalse_negatives9Adam/dense_features/movieId_embedding/embedding_weights/m:Adam/dense_features_1/userId_embedding/embedding_weights/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/m9Adam/dense_features/movieId_embedding/embedding_weights/v:Adam/dense_features_1/userId_embedding/embedding_weights/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_303970∙ч
П
╣
A__inference_model_layer_call_and_return_conditional_losses_302700

inputs
inputs_1(
dense_features_302677:	щ
+
dense_features_1_302680:
▒ъ

dense_302684:

dense_302686:
 
dense_1_302689:


dense_1_302691:
 
dense_2_302694:

dense_2_302696:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв&dense_features/StatefulPartitionedCallв(dense_features_1/StatefulPartitionedCall·
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_302677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302643А
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_1_302680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302541Ы
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_302334В
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_302684dense_302686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302347М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_302689dense_1_302691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302364О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_302694dense_2_302696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302381w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ■
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
Щ

Ї
C__inference_dense_2_layer_call_and_return_conditional_losses_303690

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╕
q
G__inference_concatenate_layer_call_and_return_conditional_losses_302334

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs:OK
'
_output_shapes
:         

 
_user_specified_nameinputs
и
X
,__inference_concatenate_layer_call_fn_303623
inputs_0
inputs_1
identity┐
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_302334`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
└
s
G__inference_concatenate_layer_call_and_return_conditional_losses_303630
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         
:         
:Q M
'
_output_shapes
:         

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         

"
_user_specified_name
inputs/1
│|
┴
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302323
features

features_1m
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302283:
▒ъ

identityИвRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupj
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
userId_embedding/ExpandDims
ExpandDims
features_1(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         z
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Й
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╧
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Р
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	О
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:а
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:y
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╦
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: |
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :y
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : └
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ▄
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:Ы
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ч
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:         z
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R В
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         Э
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         К
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         х
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         л
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Е
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ╨
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ж
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▄
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЄ
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ┴
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302283LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302283*'
_output_shapes
:         
*
dtype0ф
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302283*'
_output_shapes
:         
Б
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
н
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
К
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         й
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:З
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :у
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:р
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
┐
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
д
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
г
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:е
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:У
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Й
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         л
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:w
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : в
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:▀
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
В
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:n
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
и
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:╝
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         n
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Ы
NoOpNoOpS^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2и
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:M I
#
_output_shapes
:         
"
_user_specified_name
features:MI
#
_output_shapes
:         
"
_user_specified_name
features
Й

╟
$__inference_signature_wrapper_302825
movieid

userid
unknown:	щ

	unknown_0:
▒ъ

	unknown_1:

	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identityИвStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_302140o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
╒

╫
&__inference_model_layer_call_fn_302869
inputs_movieid
inputs_userid
unknown:	щ

	unknown_0:
▒ъ

	unknown_1:

	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinputs_movieidinputs_useridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302700o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:         
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:         
'
_user_specified_nameinputs/userId
╬
ж
1__inference_dense_features_1_layer_call_fn_303447
features_movieid
features_userid
unknown:
▒ъ

identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
╔
г
/__inference_dense_features_layer_call_fn_303261
features_movieid
features_userid
unknown:	щ

identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
йW
Б
__inference__traced_save_303837
file_prefixQ
Msavev2_dense_features_movieid_embedding_embedding_weights_read_readvariableopR
Nsavev2_dense_features_1_userid_embedding_embedding_weights_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_1_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopX
Tsavev2_adam_dense_features_movieid_embedding_embedding_weights_m_read_readvariableopY
Usavev2_adam_dense_features_1_userid_embedding_embedding_weights_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopX
Tsavev2_adam_dense_features_movieid_embedding_embedding_weights_v_read_readvariableopY
Usavev2_adam_dense_features_1_userid_embedding_embedding_weights_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: т
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*Л
valueБB■*BTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┴
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╩
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Msavev2_dense_features_movieid_embedding_embedding_weights_read_readvariableopNsavev2_dense_features_1_userid_embedding_embedding_weights_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableopTsavev2_adam_dense_features_movieid_embedding_embedding_weights_m_read_readvariableopUsavev2_adam_dense_features_1_userid_embedding_embedding_weights_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopTsavev2_adam_dense_features_movieid_embedding_embedding_weights_v_read_readvariableopUsavev2_adam_dense_features_1_userid_embedding_embedding_weights_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╕
_input_shapesж
г: :	щ
:
▒ъ
:
:
:

:
:
:: : : : : : : : : :╚:╚:╚:╚:╚:╚:╚:╚:	щ
:
▒ъ
:
:
:

:
:
::	щ
:
▒ъ
:
:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	щ
:&"
 
_output_shapes
:
▒ъ
:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:%!

_output_shapes
:	щ
:&"
 
_output_shapes
:
▒ъ
:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$  

_output_shapes

:
: !

_output_shapes
::%"!

_output_shapes
:	щ
:&#"
 
_output_shapes
:
▒ъ
:$$ 

_output_shapes

:
: %

_output_shapes
:
:$& 

_output_shapes

:

: '

_output_shapes
:
:$( 

_output_shapes

:
: )

_output_shapes
::*

_output_shapes
: 
·~
╧
J__inference_dense_features_layer_call_and_return_conditional_losses_303346
features_movieid
features_useridn
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303306:	щ

identityИвTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupk
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         Щ
movieId_embedding/ExpandDims
ExpandDimsfeatures_movieid)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         {
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Л
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╥
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Т
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Р
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Б
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:з
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: т
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:б
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ъ
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:         |
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R И
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         б
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         М
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ы
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         п
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:З
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ┌
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         и
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЎ
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ╔
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303306NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303306*'
_output_shapes
:         
*
dtype0ъ
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303306*'
_output_shapes
:         
Е
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
╡
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
М
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       г
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         н
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Й
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :щ
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:ц
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
├
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
м
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ж
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: В
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:н
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:Ч
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Л
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         │
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : к
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:х
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
Е
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
л
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:└
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         o
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOpU^movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2м
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
Щ

Ї
C__inference_dense_2_layer_call_and_return_conditional_losses_302381

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╓═
И	
!__inference__wrapped_model_302140
movieid

useridГ
pmodel_dense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_301997:	щ
Д
pmodel_dense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302077:
▒ъ
<
*model_dense_matmul_readvariableop_resource:
9
+model_dense_biasadd_readvariableop_resource:
>
,model_dense_1_matmul_readvariableop_resource:

;
-model_dense_1_biasadd_readvariableop_resource:
>
,model_dense_2_matmul_readvariableop_resource:
;
-model_dense_2_biasadd_readvariableop_resource:
identityИв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв#model/dense_2/MatMul/ReadVariableOpвimodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupвimodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupА
5model/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ║
1model/dense_features/movieId_embedding/ExpandDims
ExpandDimsmovieid>model/dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Р
Emodel/dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         Й
?model/dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqual:model/dense_features/movieId_embedding/ExpandDims:output:0Nmodel/dense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         ╡
>model/dense_features/movieId_embedding/to_sparse_input/indicesWhereCmodel/dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         С
=model/dense_features/movieId_embedding/to_sparse_input/valuesGatherNd:model/dense_features/movieId_embedding/ExpandDims:output:0Fmodel/dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ╝
Bmodel/dense_features/movieId_embedding/to_sparse_input/dense_shapeShape:model/dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	║
-model/dense_features/movieId_embedding/valuesCastFmodel/dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Ц
Lmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: Х
Kmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:√
Fmodel/dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceKmodel/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Umodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0Tmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:Р
Fmodel/dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: Р
Emodel/dense_features/movieId_embedding/movieId_embedding_weights/ProdProdOmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Omodel/dense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: У
Qmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :Р
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
Imodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Kmodel/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Zmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: б
Gmodel/dense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackNmodel/dense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Rmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:ї
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshapeFmodel/dense_features/movieId_embedding/to_sparse_input/indices:index:0Kmodel/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Pmodel/dense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :─
Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity1model/dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:         С
Omodel/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╟
Mmodel/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqual`model/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Xmodel/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ╦
Fmodel/dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereQmodel/dense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         б
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         к
Hmodel/dense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeNmodel/dense_features/movieId_embedding/movieId_embedding_weights/Where:index:0Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         Т
Pmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Kmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2_model/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Qmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Ymodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         Т
Pmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕
Kmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2`model/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Qmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Ymodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ┘
Imodel/dense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentity]model/dense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ь
Zmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ├
hmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0Tmodel/dense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Rmodel/dense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0cmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╜
lmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ┐
nmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ┐
nmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╧
fmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceymodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0umodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0wmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0wmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskа
_model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquexmodel/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Ю
imodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherpmodel_dense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_301997cmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*Г
_classy
wuloc:@model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/301997*'
_output_shapes
:         
*
dtype0к
rmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*Г
_classy
wuloc:@model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/301997*'
_output_shapes
:         
п
tmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
Й
Xmodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMean}model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0emodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0omodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
б
Pmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
Jmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshape~model/dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Ymodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╫
Fmodel/dense_features/movieId_embedding/movieId_embedding_weights/ShapeShapeamodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ю
Tmodel/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:а
Vmodel/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
Vmodel/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceOmodel/dense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0]model/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0_model/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0_model/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
Hmodel/dense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :и
Fmodel/dense_features/movieId_embedding/movieId_embedding_weights/stackPackQmodel/dense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:е
Emodel/dense_features/movieId_embedding/movieId_embedding_weights/TileTileSmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Omodel/dense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
э
Kmodel/dense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeamodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
А
@model/dense_features/movieId_embedding/movieId_embedding_weightsSelectNmodel/dense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Omodel/dense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0amodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
╨
Gmodel/dense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastKmodel/dense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Ш
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: Ч
Mmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:Б
Hmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceKmodel/dense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0Vmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:┴
Hmodel/dense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapeImodel/dense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Ш
Nmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:а
Mmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         З
Hmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceQmodel/dense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0Wmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0Vmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:О
Lmodel/dense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ■
Gmodel/dense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Qmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Qmodel/dense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0Umodel/dense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:д
Jmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapeImodel/dense_features/movieId_embedding/movieId_embedding_weights:output:0Pmodel/dense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
п
,model/dense_features/movieId_embedding/ShapeShapeSmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:Д
:model/dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ж
<model/dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<model/dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
4model/dense_features/movieId_embedding/strided_sliceStridedSlice5model/dense_features/movieId_embedding/Shape:output:0Cmodel/dense_features/movieId_embedding/strided_slice/stack:output:0Emodel/dense_features/movieId_embedding/strided_slice/stack_1:output:0Emodel/dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6model/dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ъ
4model/dense_features/movieId_embedding/Reshape/shapePack=model/dense_features/movieId_embedding/strided_slice:output:0?model/dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
: 
.model/dense_features/movieId_embedding/ReshapeReshapeSmodel/dense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0=model/dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
q
&model/dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Щ
"model/dense_features/concat/concatIdentity7model/dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
Б
6model/dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╗
2model/dense_features_1/userId_embedding/ExpandDims
ExpandDimsuserid?model/dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         С
Fmodel/dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         М
@model/dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqual;model/dense_features_1/userId_embedding/ExpandDims:output:0Omodel/dense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         ╖
?model/dense_features_1/userId_embedding/to_sparse_input/indicesWhereDmodel/dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         Ф
>model/dense_features_1/userId_embedding/to_sparse_input/valuesGatherNd;model/dense_features_1/userId_embedding/ExpandDims:output:0Gmodel/dense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ╛
Cmodel/dense_features_1/userId_embedding/to_sparse_input/dense_shapeShape;model/dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	╝
.model/dense_features_1/userId_embedding/valuesCastGmodel/dense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Ц
Lmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: Х
Kmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:№
Fmodel/dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceLmodel/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Umodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0Tmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:Р
Fmodel/dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: Р
Emodel/dense_features_1/userId_embedding/userId_embedding_weights/ProdProdOmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Omodel/dense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: У
Qmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :Р
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Imodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Lmodel/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Zmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: б
Gmodel/dense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackNmodel/dense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Rmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:ў
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeGmodel/dense_features_1/userId_embedding/to_sparse_input/indices:index:0Lmodel/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Pmodel/dense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :┼
Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity2model/dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:         С
Omodel/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╟
Mmodel/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqual`model/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Xmodel/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ╦
Fmodel/dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereQmodel/dense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         б
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         к
Hmodel/dense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeNmodel/dense_features_1/userId_embedding/userId_embedding_weights/Where:index:0Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         Т
Pmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Kmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2_model/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Qmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Ymodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         Т
Pmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕
Kmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2`model/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Qmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Ymodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ┘
Imodel/dense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentity]model/dense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ь
Zmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ├
hmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0Tmodel/dense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Rmodel/dense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0cmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╜
lmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ┐
nmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ┐
nmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╧
fmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceymodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0umodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0wmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0wmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskа
_model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquexmodel/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Ю
imodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherpmodel_dense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302077cmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*Г
_classy
wuloc:@model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302077*'
_output_shapes
:         
*
dtype0к
rmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*Г
_classy
wuloc:@model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302077*'
_output_shapes
:         
п
tmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
Й
Xmodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMean}model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0emodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0omodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
б
Pmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
Jmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshape~model/dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Ymodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╫
Fmodel/dense_features_1/userId_embedding/userId_embedding_weights/ShapeShapeamodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ю
Tmodel/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:а
Vmodel/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
Vmodel/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceOmodel/dense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0]model/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0_model/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0_model/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
Hmodel/dense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :и
Fmodel/dense_features_1/userId_embedding/userId_embedding_weights/stackPackQmodel/dense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:е
Emodel/dense_features_1/userId_embedding/userId_embedding_weights/TileTileSmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Omodel/dense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
э
Kmodel/dense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeamodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
А
@model/dense_features_1/userId_embedding/userId_embedding_weightsSelectNmodel/dense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Omodel/dense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0amodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
╤
Gmodel/dense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastLmodel/dense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Ш
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: Ч
Mmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:Б
Hmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceKmodel/dense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0Vmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:┴
Hmodel/dense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapeImodel/dense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Ш
Nmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:а
Mmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         З
Hmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceQmodel/dense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0Wmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0Vmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:О
Lmodel/dense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ■
Gmodel/dense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Qmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Qmodel/dense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0Umodel/dense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:д
Jmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapeImodel/dense_features_1/userId_embedding/userId_embedding_weights:output:0Pmodel/dense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
░
-model/dense_features_1/userId_embedding/ShapeShapeSmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:Е
;model/dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
=model/dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=model/dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
5model/dense_features_1/userId_embedding/strided_sliceStridedSlice6model/dense_features_1/userId_embedding/Shape:output:0Dmodel/dense_features_1/userId_embedding/strided_slice/stack:output:0Fmodel/dense_features_1/userId_embedding/strided_slice/stack_1:output:0Fmodel/dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7model/dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
э
5model/dense_features_1/userId_embedding/Reshape/shapePack>model/dense_features_1/userId_embedding/strided_slice:output:0@model/dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
/model/dense_features_1/userId_embedding/ReshapeReshapeSmodel/dense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0>model/dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
s
(model/dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ь
$model/dense_features_1/concat/concatIdentity8model/dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :у
model/concatenate/concatConcatV2+model/dense_features/concat/concat:output:0-model/dense_features_1/concat/concat:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         М
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ь
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
К
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         
Р
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Э
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
О
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0а
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
Р
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Я
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         О
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitymodel/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOpj^model/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupj^model/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2╓
imodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupimodel/dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup2╓
imodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupimodel/dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
Н
╕
A__inference_model_layer_call_and_return_conditional_losses_302795
movieid

userid(
dense_features_302772:	щ
+
dense_features_1_302775:
▒ъ

dense_302779:

dense_302781:
 
dense_1_302784:


dense_1_302786:
 
dense_2_302789:

dense_2_302791:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв&dense_features/StatefulPartitionedCallв(dense_features_1/StatefulPartitionedCall∙
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_302772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302643 
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_1_302775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302541Ы
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_302334В
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_302779dense_302781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302347М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_302784dense_1_302786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302364О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_302789dense_2_302791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302381w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ■
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
╓~
┬
J__inference_dense_features_layer_call_and_return_conditional_losses_302643
features

features_1n
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302603:	щ

identityИвTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupk
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
movieId_embedding/ExpandDims
ExpandDimsfeatures)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         {
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Л
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╥
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Т
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Р
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Б
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:з
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: т
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:б
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ъ
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:         |
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R И
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         б
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         М
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ы
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         п
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:З
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ┌
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         и
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЎ
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ╔
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302603NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302603*'
_output_shapes
:         
*
dtype0ъ
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302603*'
_output_shapes
:         
Е
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
╡
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
М
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       г
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         н
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Й
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :щ
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:ц
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
├
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
м
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ж
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: В
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:н
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:Ч
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Л
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         │
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : к
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:х
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
Е
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
л
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:└
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         o
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOpU^movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2м
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:M I
#
_output_shapes
:         
"
_user_specified_name
features:MI
#
_output_shapes
:         
"
_user_specified_name
features
Е╗
╘
A__inference_model_layer_call_and_return_conditional_losses_303057
inputs_movieid
inputs_userid}
jdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302914:	щ
~
jdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302994:
▒ъ
6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвcdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupвcdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupz
/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╡
+dense_features/movieId_embedding/ExpandDims
ExpandDimsinputs_movieid8dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         К
?dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ў
9dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqual4dense_features/movieId_embedding/ExpandDims:output:0Hdense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         й
8dense_features/movieId_embedding/to_sparse_input/indicesWhere=dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:          
7dense_features/movieId_embedding/to_sparse_input/valuesGatherNd4dense_features/movieId_embedding/ExpandDims:output:0@dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ░
<dense_features/movieId_embedding/to_sparse_input/dense_shapeShape4dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	о
'dense_features/movieId_embedding/valuesCast@dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Р
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: П
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:у
@dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Odense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:К
@dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
?dense_features/movieId_embedding/movieId_embedding_weights/ProdProdIdense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Idense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: Н
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :К
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Tdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: П
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackHdense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:▌
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape@dense_features/movieId_embedding/to_sparse_input/indices:index:0Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :╕
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity+dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:         Л
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╡
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualZdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Rdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ┐
@dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereKdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         Ы
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ш
Bdense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeHdense_features/movieId_embedding/movieId_embedding_weights/Where:index:0Qdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         М
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2Ydense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         М
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : а
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2Zdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ═
Cdense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentityWdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ц
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R е
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0]dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╖
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ╣
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ╣
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▒
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskФ
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Е
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302914]dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302914*'
_output_shapes
:         
*
dtype0Ч
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302914*'
_output_shapes
:         
г
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
ё
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
Ы
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshapexdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╦
@dense_features/movieId_embedding/movieId_embedding_weights/ShapeShape[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ш
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceIdense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0Wdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :Ц
@dense_features/movieId_embedding/movieId_embedding_weights/stackPackKdense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:У
?dense_features/movieId_embedding/movieId_embedding_weights/TileTileMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Idense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
с
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLike[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ш
:dense_features/movieId_embedding/movieId_embedding_weightsSelectHdense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Idense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
─
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Т
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: С
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:щ
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceEdense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:╡
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Т
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         я
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceKdense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:И
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ц
Adense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0Odense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:Т
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
г
&dense_features/movieId_embedding/ShapeShapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:~
4dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.dense_features/movieId_embedding/strided_sliceStridedSlice/dense_features/movieId_embedding/Shape:output:0=dense_features/movieId_embedding/strided_slice/stack:output:0?dense_features/movieId_embedding/strided_slice/stack_1:output:0?dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
╪
.dense_features/movieId_embedding/Reshape/shapePack7dense_features/movieId_embedding/strided_slice:output:09dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:э
(dense_features/movieId_embedding/ReshapeReshapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:07dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
k
 dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
dense_features/concat/concatIdentity1dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
{
0dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╢
,dense_features_1/userId_embedding/ExpandDims
ExpandDimsinputs_userid9dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Л
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ·
:dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqual5dense_features_1/userId_embedding/ExpandDims:output:0Idense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         л
9dense_features_1/userId_embedding/to_sparse_input/indicesWhere>dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         В
8dense_features_1/userId_embedding/to_sparse_input/valuesGatherNd5dense_features_1/userId_embedding/ExpandDims:output:0Adense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ▓
=dense_features_1/userId_embedding/to_sparse_input/dense_shapeShape5dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	░
(dense_features_1/userId_embedding/valuesCastAdense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Р
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: П
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:ф
@dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Odense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:К
@dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
?dense_features_1/userId_embedding/userId_embedding_weights/ProdProdIdense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Idense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: Н
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :К
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Tdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: П
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackHdense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:▀
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeAdense_features_1/userId_embedding/to_sparse_input/indices:index:0Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :╣
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity,dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:         Л
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╡
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualZdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Rdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ┐
@dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereKdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         Ы
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ш
Bdense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeHdense_features_1/userId_embedding/userId_embedding_weights/Where:index:0Qdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         М
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2Ydense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         М
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : а
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2Zdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ═
Cdense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentityWdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ц
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R е
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0]dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╖
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ╣
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ╣
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▒
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskФ
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Е
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302994]dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302994*'
_output_shapes
:         
*
dtype0Ч
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302994*'
_output_shapes
:         
г
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
ё
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
Ы
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshapexdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╦
@dense_features_1/userId_embedding/userId_embedding_weights/ShapeShape[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ш
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceIdense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0Wdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :Ц
@dense_features_1/userId_embedding/userId_embedding_weights/stackPackKdense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:У
?dense_features_1/userId_embedding/userId_embedding_weights/TileTileMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Idense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
с
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLike[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ш
:dense_features_1/userId_embedding/userId_embedding_weightsSelectHdense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Idense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
┼
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Т
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: С
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:щ
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceEdense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:╡
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Т
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         я
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceKdense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:И
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ц
Adense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0Odense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:Т
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
д
'dense_features_1/userId_embedding/ShapeShapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:
5dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/dense_features_1/userId_embedding/strided_sliceStridedSlice0dense_features_1/userId_embedding/Shape:output:0>dense_features_1/userId_embedding/strided_slice/stack:output:0@dense_features_1/userId_embedding/strided_slice/stack_1:output:0@dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
█
/dense_features_1/userId_embedding/Reshape/shapePack8dense_features_1/userId_embedding/strided_slice:output:0:dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:я
)dense_features_1/userId_embedding/ReshapeReshapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:08dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
m
"dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Р
dense_features_1/concat/concatIdentity2dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╦
concatenate/concatConcatV2%dense_features/concat/concat:output:0'dense_features_1/concat/concat:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0К
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         
Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         ╤
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpd^dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupd^dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2╩
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupcdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup2╩
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupcdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:S O
#
_output_shapes
:         
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:         
'
_user_specified_nameinputs/userId
│|
┴
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302541
features

features_1m
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302501:
▒ъ

identityИвRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupj
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
userId_embedding/ExpandDims
ExpandDims
features_1(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         z
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Й
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╧
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Р
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	О
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:а
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:y
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╦
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: |
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :y
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : └
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ▄
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:Ы
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ч
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:         z
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R В
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         Э
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         К
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         х
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         л
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Е
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ╨
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ж
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▄
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЄ
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ┴
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302501LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302501*'
_output_shapes
:         
*
dtype0ф
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302501*'
_output_shapes
:         
Б
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
н
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
К
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         й
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:З
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :у
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:р
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
┐
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
д
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
г
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:е
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:У
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Й
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         л
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:w
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : в
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:▀
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
В
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:n
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
и
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:╝
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         n
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Ы
NoOpNoOpS^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2и
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:M I
#
_output_shapes
:         
"
_user_specified_name
features:MI
#
_output_shapes
:         
"
_user_specified_name
features
╘|
╬
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303532
features_movieid
features_useridm
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303492:
▒ъ

identityИвRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupj
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
userId_embedding/ExpandDims
ExpandDimsfeatures_userid(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         z
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Й
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╧
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Р
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	О
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:а
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:y
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╦
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: |
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :y
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : └
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ▄
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:Ы
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ч
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:         z
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R В
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         Э
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         К
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         х
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         л
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Е
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ╨
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ж
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▄
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЄ
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ┴
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303492LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303492*'
_output_shapes
:         
*
dtype0ф
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303492*'
_output_shapes
:         
Б
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
н
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
К
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         й
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:З
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :у
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:р
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
┐
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
д
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
г
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:е
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:У
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Й
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         л
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:w
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : в
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:▀
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
В
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:n
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
и
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:╝
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         n
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Ы
NoOpNoOpS^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2и
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
Ш

Є
A__inference_dense_layer_call_and_return_conditional_losses_302347

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╘|
╬
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303617
features_movieid
features_useridm
Yuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303577:
▒ъ

identityИвRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupj
userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
userId_embedding/ExpandDims
ExpandDimsfeatures_userid(userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         z
/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
)userId_embedding/to_sparse_input/NotEqualNotEqual$userId_embedding/ExpandDims:output:08userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Й
(userId_embedding/to_sparse_input/indicesWhere-userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╧
'userId_embedding/to_sparse_input/valuesGatherNd$userId_embedding/ExpandDims:output:00userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Р
,userId_embedding/to_sparse_input/dense_shapeShape$userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	О
userId_embedding/valuesCast0userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         
5userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:а
/userId_embedding/userId_embedding_weights/SliceSlice5userId_embedding/to_sparse_input/dense_shape:output:0>userId_embedding/userId_embedding_weights/Slice/begin:output:0=userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:y
/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╦
.userId_embedding/userId_embedding_weights/ProdProd8userId_embedding/userId_embedding_weights/Slice:output:08userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: |
:userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :y
7userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : └
2userId_embedding/userId_embedding_weights/GatherV2GatherV25userId_embedding/to_sparse_input/dense_shape:output:0CuserId_embedding/userId_embedding_weights/GatherV2/indices:output:0@userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ▄
0userId_embedding/userId_embedding_weights/Cast/xPack7userId_embedding/userId_embedding_weights/Prod:output:0;userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:Ы
7userId_embedding/userId_embedding_weights/SparseReshapeSparseReshape0userId_embedding/to_sparse_input/indices:index:05userId_embedding/to_sparse_input/dense_shape:output:09userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ч
@userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentityuserId_embedding/values:y:0*
T0	*#
_output_shapes
:         z
8userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R В
6userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualIuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0AuserId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         Э
/userId_embedding/userId_embedding_weights/WhereWhere:userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         К
7userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         х
1userId_embedding/userId_embedding_weights/ReshapeReshape7userId_embedding/userId_embedding_weights/Where:index:0@userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
4userId_embedding/userId_embedding_weights/GatherV2_1GatherV2HuserId_embedding/userId_embedding_weights/SparseReshape:output_indices:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         {
9userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
4userId_embedding/userId_embedding_weights/GatherV2_2GatherV2IuserId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0:userId_embedding/userId_embedding_weights/Reshape:output:0BuserId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         л
2userId_embedding/userId_embedding_weights/IdentityIdentityFuserId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Е
CuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ╨
QuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows=userId_embedding/userId_embedding_weights/GatherV2_1:output:0=userId_embedding/userId_embedding_weights/GatherV2_2:output:0;userId_embedding/userId_embedding_weights/Identity:output:0LuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ж
UuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       и
WuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▄
OuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicebuserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0`userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЄ
HuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniqueauserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ┴
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherYuserid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303577LuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303577*'
_output_shapes
:         
*
dtype0ф
[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity[userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*l
_classb
`^loc:@userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303577*'
_output_shapes
:         
Б
]userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityduserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
н
AuserId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0NuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0XuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
К
9userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
3userId_embedding/userId_embedding_weights/Reshape_1ReshapeguserId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0BuserId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         й
/userId_embedding/userId_embedding_weights/ShapeShapeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:З
=userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7userId_embedding/userId_embedding_weights/strided_sliceStridedSlice8userId_embedding/userId_embedding_weights/Shape:output:0FuserId_embedding/userId_embedding_weights/strided_slice/stack:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0HuserId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :у
/userId_embedding/userId_embedding_weights/stackPack:userId_embedding/userId_embedding_weights/stack/0:output:0@userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:р
.userId_embedding/userId_embedding_weights/TileTile<userId_embedding/userId_embedding_weights/Reshape_1:output:08userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
┐
4userId_embedding/userId_embedding_weights/zeros_like	ZerosLikeJuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
д
)userId_embedding/userId_embedding_weightsSelect7userId_embedding/userId_embedding_weights/Tile:output:08userId_embedding/userId_embedding_weights/zeros_like:y:0JuserId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
г
0userId_embedding/userId_embedding_weights/Cast_1Cast5userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:е
1userId_embedding/userId_embedding_weights/Slice_1Slice4userId_embedding/userId_embedding_weights/Cast_1:y:0@userId_embedding/userId_embedding_weights/Slice_1/begin:output:0?userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:У
1userId_embedding/userId_embedding_weights/Shape_1Shape2userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Б
7userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Й
6userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         л
1userId_embedding/userId_embedding_weights/Slice_2Slice:userId_embedding/userId_embedding_weights/Shape_1:output:0@userId_embedding/userId_embedding_weights/Slice_2/begin:output:0?userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:w
5userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : в
0userId_embedding/userId_embedding_weights/concatConcatV2:userId_embedding/userId_embedding_weights/Slice_1:output:0:userId_embedding/userId_embedding_weights/Slice_2:output:0>userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:▀
3userId_embedding/userId_embedding_weights/Reshape_2Reshape2userId_embedding/userId_embedding_weights:output:09userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
В
userId_embedding/ShapeShape<userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:n
$userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
userId_embedding/strided_sliceStridedSliceuserId_embedding/Shape:output:0-userId_embedding/strided_slice/stack:output:0/userId_embedding/strided_slice/stack_1:output:0/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
и
userId_embedding/Reshape/shapePack'userId_embedding/strided_slice:output:0)userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:╝
userId_embedding/ReshapeReshape<userId_embedding/userId_embedding_weights/Reshape_2:output:0'userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         n
concat/concatIdentity!userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Ы
NoOpNoOpS^userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2и
RuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupRuserId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
╓~
┬
J__inference_dense_features_layer_call_and_return_conditional_losses_302234
features

features_1n
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302194:	щ

identityИвTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupk
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
movieId_embedding/ExpandDims
ExpandDimsfeatures)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         {
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Л
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╥
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Т
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Р
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Б
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:з
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: т
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:б
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ъ
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:         |
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R И
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         б
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         М
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ы
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         п
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:З
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ┌
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         и
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЎ
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ╔
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_302194NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302194*'
_output_shapes
:         
*
dtype0ъ
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/302194*'
_output_shapes
:         
Е
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
╡
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
М
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       г
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         н
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Й
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :щ
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:ц
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
├
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
м
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ж
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: В
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:н
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:Ч
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Л
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         │
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : к
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:х
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
Е
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
л
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:└
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         o
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOpU^movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2м
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:M I
#
_output_shapes
:         
"
_user_specified_name
features:MI
#
_output_shapes
:         
"
_user_specified_name
features
└
Х
(__inference_dense_2_layer_call_fn_303679

inputs
unknown:

	unknown_0:
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302381o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Н
╕
A__inference_model_layer_call_and_return_conditional_losses_302768
movieid

userid(
dense_features_302745:	щ
+
dense_features_1_302748:
▒ъ

dense_302752:

dense_302754:
 
dense_1_302757:


dense_1_302759:
 
dense_2_302762:

dense_2_302764:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв&dense_features/StatefulPartitionedCallв(dense_features_1/StatefulPartitionedCall∙
&dense_features/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_302745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302234 
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallmovieiduseriddense_features_1_302748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302323Ы
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_302334В
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_302752dense_302754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302347М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_302757dense_1_302759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302364О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_302762dense_2_302764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302381w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ■
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
Е╗
╘
A__inference_model_layer_call_and_return_conditional_losses_303245
inputs_movieid
inputs_userid}
jdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303102:	щ
~
jdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303182:
▒ъ
6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвcdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupвcdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupz
/dense_features/movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╡
+dense_features/movieId_embedding/ExpandDims
ExpandDimsinputs_movieid8dense_features/movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         К
?dense_features/movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ў
9dense_features/movieId_embedding/to_sparse_input/NotEqualNotEqual4dense_features/movieId_embedding/ExpandDims:output:0Hdense_features/movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         й
8dense_features/movieId_embedding/to_sparse_input/indicesWhere=dense_features/movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:          
7dense_features/movieId_embedding/to_sparse_input/valuesGatherNd4dense_features/movieId_embedding/ExpandDims:output:0@dense_features/movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ░
<dense_features/movieId_embedding/to_sparse_input/dense_shapeShape4dense_features/movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	о
'dense_features/movieId_embedding/valuesCast@dense_features/movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Р
Fdense_features/movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: П
Edense_features/movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:у
@dense_features/movieId_embedding/movieId_embedding_weights/SliceSliceEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Odense_features/movieId_embedding/movieId_embedding_weights/Slice/begin:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:К
@dense_features/movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
?dense_features/movieId_embedding/movieId_embedding_weights/ProdProdIdense_features/movieId_embedding/movieId_embedding_weights/Slice:output:0Idense_features/movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: Н
Kdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :К
Hdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Cdense_features/movieId_embedding/movieId_embedding_weights/GatherV2GatherV2Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Tdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: П
Adense_features/movieId_embedding/movieId_embedding_weights/Cast/xPackHdense_features/movieId_embedding/movieId_embedding_weights/Prod:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:▌
Hdense_features/movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape@dense_features/movieId_embedding/to_sparse_input/indices:index:0Edense_features/movieId_embedding/to_sparse_input/dense_shape:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :╕
Qdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentity+dense_features/movieId_embedding/values:y:0*
T0	*#
_output_shapes
:         Л
Idense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╡
Gdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualZdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Rdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ┐
@dense_features/movieId_embedding/movieId_embedding_weights/WhereWhereKdense_features/movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         Ы
Hdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ш
Bdense_features/movieId_embedding/movieId_embedding_weights/ReshapeReshapeHdense_features/movieId_embedding/movieId_embedding_weights/Where:index:0Qdense_features/movieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         М
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2Ydense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         М
Jdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : а
Edense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2Zdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Reshape:output:0Sdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ═
Cdense_features/movieId_embedding/movieId_embedding_weights/IdentityIdentityWdense_features/movieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ц
Tdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R е
bdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0Ndense_features/movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0Ldense_features/movieId_embedding/movieId_embedding_weights/Identity:output:0]dense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╖
fdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ╣
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ╣
hdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▒
`dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskФ
Ydense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Е
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303102]dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303102*'
_output_shapes
:         
*
dtype0Ч
ldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303102*'
_output_shapes
:         
г
ndense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
ё
Rdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
Ы
Jdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_1Reshapexdense_features/movieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╦
@dense_features/movieId_embedding/movieId_embedding_weights/ShapeShape[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ш
Ndense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
Hdense_features/movieId_embedding/movieId_embedding_weights/strided_sliceStridedSliceIdense_features/movieId_embedding/movieId_embedding_weights/Shape:output:0Wdense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0Ydense_features/movieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
Bdense_features/movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :Ц
@dense_features/movieId_embedding/movieId_embedding_weights/stackPackKdense_features/movieId_embedding/movieId_embedding_weights/stack/0:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:У
?dense_features/movieId_embedding/movieId_embedding_weights/TileTileMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_1:output:0Idense_features/movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
с
Edense_features/movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLike[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ш
:dense_features/movieId_embedding/movieId_embedding_weightsSelectHdense_features/movieId_embedding/movieId_embedding_weights/Tile:output:0Idense_features/movieId_embedding/movieId_embedding_weights/zeros_like:y:0[dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
─
Adense_features/movieId_embedding/movieId_embedding_weights/Cast_1CastEdense_features/movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Т
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: С
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:щ
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_1SliceEdense_features/movieId_embedding/movieId_embedding_weights/Cast_1:y:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:╡
Bdense_features/movieId_embedding/movieId_embedding_weights/Shape_1ShapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Т
Hdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Gdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         я
Bdense_features/movieId_embedding/movieId_embedding_weights/Slice_2SliceKdense_features/movieId_embedding/movieId_embedding_weights/Shape_1:output:0Qdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0Pdense_features/movieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:И
Fdense_features/movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ц
Adense_features/movieId_embedding/movieId_embedding_weights/concatConcatV2Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_1:output:0Kdense_features/movieId_embedding/movieId_embedding_weights/Slice_2:output:0Odense_features/movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:Т
Ddense_features/movieId_embedding/movieId_embedding_weights/Reshape_2ReshapeCdense_features/movieId_embedding/movieId_embedding_weights:output:0Jdense_features/movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
г
&dense_features/movieId_embedding/ShapeShapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:~
4dense_features/movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6dense_features/movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6dense_features/movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.dense_features/movieId_embedding/strided_sliceStridedSlice/dense_features/movieId_embedding/Shape:output:0=dense_features/movieId_embedding/strided_slice/stack:output:0?dense_features/movieId_embedding/strided_slice/stack_1:output:0?dense_features/movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
╪
.dense_features/movieId_embedding/Reshape/shapePack7dense_features/movieId_embedding/strided_slice:output:09dense_features/movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:э
(dense_features/movieId_embedding/ReshapeReshapeMdense_features/movieId_embedding/movieId_embedding_weights/Reshape_2:output:07dense_features/movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
k
 dense_features/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
dense_features/concat/concatIdentity1dense_features/movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
{
0dense_features_1/userId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╢
,dense_features_1/userId_embedding/ExpandDims
ExpandDimsinputs_userid9dense_features_1/userId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Л
@dense_features_1/userId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ·
:dense_features_1/userId_embedding/to_sparse_input/NotEqualNotEqual5dense_features_1/userId_embedding/ExpandDims:output:0Idense_features_1/userId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         л
9dense_features_1/userId_embedding/to_sparse_input/indicesWhere>dense_features_1/userId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         В
8dense_features_1/userId_embedding/to_sparse_input/valuesGatherNd5dense_features_1/userId_embedding/ExpandDims:output:0Adense_features_1/userId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         ▓
=dense_features_1/userId_embedding/to_sparse_input/dense_shapeShape5dense_features_1/userId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	░
(dense_features_1/userId_embedding/valuesCastAdense_features_1/userId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Р
Fdense_features_1/userId_embedding/userId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: П
Edense_features_1/userId_embedding/userId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:ф
@dense_features_1/userId_embedding/userId_embedding_weights/SliceSliceFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Odense_features_1/userId_embedding/userId_embedding_weights/Slice/begin:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:К
@dense_features_1/userId_embedding/userId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
?dense_features_1/userId_embedding/userId_embedding_weights/ProdProdIdense_features_1/userId_embedding/userId_embedding_weights/Slice:output:0Idense_features_1/userId_embedding/userId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: Н
Kdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :К
Hdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Cdense_features_1/userId_embedding/userId_embedding_weights/GatherV2GatherV2Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Tdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/indices:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: П
Adense_features_1/userId_embedding/userId_embedding_weights/Cast/xPackHdense_features_1/userId_embedding/userId_embedding_weights/Prod:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:▀
Hdense_features_1/userId_embedding/userId_embedding_weights/SparseReshapeSparseReshapeAdense_features_1/userId_embedding/to_sparse_input/indices:index:0Fdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :╣
Qdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/IdentityIdentity,dense_features_1/userId_embedding/values:y:0*
T0	*#
_output_shapes
:         Л
Idense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╡
Gdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqualGreaterEqualZdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Rdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         ┐
@dense_features_1/userId_embedding/userId_embedding_weights/WhereWhereKdense_features_1/userId_embedding/userId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         Ы
Hdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ш
Bdense_features_1/userId_embedding/userId_embedding_weights/ReshapeReshapeHdense_features_1/userId_embedding/userId_embedding_weights/Where:index:0Qdense_features_1/userId_embedding/userId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         М
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1GatherV2Ydense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_indices:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         М
Jdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : а
Edense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2GatherV2Zdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape/Identity:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Reshape:output:0Sdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         ═
Cdense_features_1/userId_embedding/userId_embedding_weights/IdentityIdentityWdense_features_1/userId_embedding/userId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:Ц
Tdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R е
bdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features_1/userId_embedding/userId_embedding_weights/GatherV2_1:output:0Ndense_features_1/userId_embedding/userId_embedding_weights/GatherV2_2:output:0Ldense_features_1/userId_embedding/userId_embedding_weights/Identity:output:0]dense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         ╖
fdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ╣
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ╣
hdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▒
`dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskФ
Ydense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         Е
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherjdense_features_1_userid_embedding_userid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303182]dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303182*'
_output_shapes
:         
*
dtype0Ч
ldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*}
_classs
qoloc:@dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303182*'
_output_shapes
:         
г
ndense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identityudense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
ё
Rdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0_dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
Ы
Jdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_1Reshapexdense_features_1/userId_embedding/userId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         ╦
@dense_features_1/userId_embedding/userId_embedding_weights/ShapeShape[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Ш
Ndense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Pdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
Hdense_features_1/userId_embedding/userId_embedding_weights/strided_sliceStridedSliceIdense_features_1/userId_embedding/userId_embedding_weights/Shape:output:0Wdense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_1:output:0Ydense_features_1/userId_embedding/userId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
Bdense_features_1/userId_embedding/userId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :Ц
@dense_features_1/userId_embedding/userId_embedding_weights/stackPackKdense_features_1/userId_embedding/userId_embedding_weights/stack/0:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:У
?dense_features_1/userId_embedding/userId_embedding_weights/TileTileMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_1:output:0Idense_features_1/userId_embedding/userId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
с
Edense_features_1/userId_embedding/userId_embedding_weights/zeros_like	ZerosLike[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ш
:dense_features_1/userId_embedding/userId_embedding_weightsSelectHdense_features_1/userId_embedding/userId_embedding_weights/Tile:output:0Idense_features_1/userId_embedding/userId_embedding_weights/zeros_like:y:0[dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
┼
Adense_features_1/userId_embedding/userId_embedding_weights/Cast_1CastFdense_features_1/userId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Т
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: С
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:щ
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_1SliceEdense_features_1/userId_embedding/userId_embedding_weights/Cast_1:y:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:╡
Bdense_features_1/userId_embedding/userId_embedding_weights/Shape_1ShapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0*
T0*
_output_shapes
:Т
Hdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Gdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         я
Bdense_features_1/userId_embedding/userId_embedding_weights/Slice_2SliceKdense_features_1/userId_embedding/userId_embedding_weights/Shape_1:output:0Qdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/begin:output:0Pdense_features_1/userId_embedding/userId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:И
Fdense_features_1/userId_embedding/userId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ц
Adense_features_1/userId_embedding/userId_embedding_weights/concatConcatV2Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_1:output:0Kdense_features_1/userId_embedding/userId_embedding_weights/Slice_2:output:0Odense_features_1/userId_embedding/userId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:Т
Ddense_features_1/userId_embedding/userId_embedding_weights/Reshape_2ReshapeCdense_features_1/userId_embedding/userId_embedding_weights:output:0Jdense_features_1/userId_embedding/userId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
д
'dense_features_1/userId_embedding/ShapeShapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:
5dense_features_1/userId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7dense_features_1/userId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7dense_features_1/userId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
/dense_features_1/userId_embedding/strided_sliceStridedSlice0dense_features_1/userId_embedding/Shape:output:0>dense_features_1/userId_embedding/strided_slice/stack:output:0@dense_features_1/userId_embedding/strided_slice/stack_1:output:0@dense_features_1/userId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features_1/userId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
█
/dense_features_1/userId_embedding/Reshape/shapePack8dense_features_1/userId_embedding/strided_slice:output:0:dense_features_1/userId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:я
)dense_features_1/userId_embedding/ReshapeReshapeMdense_features_1/userId_embedding/userId_embedding_weights/Reshape_2:output:08dense_features_1/userId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
m
"dense_features_1/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Р
dense_features_1/concat/concatIdentity2dense_features_1/userId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╦
concatenate/concatConcatV2%dense_features/concat/concat:output:0'dense_features_1/concat/concat:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0К
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         
Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         ╤
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpd^dense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupd^dense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2╩
cdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupcdense_features/movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup2╩
cdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookupcdense_features_1/userId_embedding/userId_embedding_weights/embedding_lookup_sparse/embedding_lookup:S O
#
_output_shapes
:         
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:         
'
_user_specified_nameinputs/userId
╝
У
&__inference_dense_layer_call_fn_303639

inputs
unknown:

	unknown_0:

identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л

╔
&__inference_model_layer_call_fn_302407
movieid

userid
unknown:	щ

	unknown_0:
▒ъ

	unknown_1:

	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302388o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
╔
г
/__inference_dense_features_layer_call_fn_303253
features_movieid
features_userid
unknown:	щ

identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
л

╔
&__inference_model_layer_call_fn_302741
movieid

userid
unknown:	щ

	unknown_0:
▒ъ

	unknown_1:

	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallmovieiduseridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302700o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:         
!
_user_specified_name	movieId:KG
#
_output_shapes
:         
 
_user_specified_nameuserId
П
╣
A__inference_model_layer_call_and_return_conditional_losses_302388

inputs
inputs_1(
dense_features_302235:	щ
+
dense_features_1_302324:
▒ъ

dense_302348:

dense_302350:
 
dense_1_302365:


dense_1_302367:
 
dense_2_302382:

dense_2_302384:
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв&dense_features/StatefulPartitionedCallв(dense_features_1/StatefulPartitionedCall·
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_302235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_302234А
(dense_features_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dense_features_1_302324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302323Ы
concatenate/PartitionedCallPartitionedCall/dense_features/StatefulPartitionedCall:output:01dense_features_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_302334В
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_302348dense_302350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_302347М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_302365dense_1_302367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302364О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_302382dense_2_302384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_302381w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ■
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall)^dense_features_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall2T
(dense_features_1/StatefulPartitionedCall(dense_features_1/StatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
·~
╧
J__inference_dense_features_layer_call_and_return_conditional_losses_303431
features_movieid
features_useridn
[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303391:	щ

identityИвTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupk
 movieId_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         Щ
movieId_embedding/ExpandDims
ExpandDimsfeatures_movieid)movieId_embedding/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         {
0movieId_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
*movieId_embedding/to_sparse_input/NotEqualNotEqual%movieId_embedding/ExpandDims:output:09movieId_embedding/to_sparse_input/ignore_value/x:output:0*
T0*'
_output_shapes
:         Л
)movieId_embedding/to_sparse_input/indicesWhere.movieId_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:         ╥
(movieId_embedding/to_sparse_input/valuesGatherNd%movieId_embedding/ExpandDims:output:01movieId_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:         Т
-movieId_embedding/to_sparse_input/dense_shapeShape%movieId_embedding/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Р
movieId_embedding/valuesCast1movieId_embedding/to_sparse_input/values:output:0*

DstT0	*

SrcT0*#
_output_shapes
:         Б
7movieId_embedding/movieId_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: А
6movieId_embedding/movieId_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:з
1movieId_embedding/movieId_embedding_weights/SliceSlice6movieId_embedding/to_sparse_input/dense_shape:output:0@movieId_embedding/movieId_embedding_weights/Slice/begin:output:0?movieId_embedding/movieId_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1movieId_embedding/movieId_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0movieId_embedding/movieId_embedding_weights/ProdProd:movieId_embedding/movieId_embedding_weights/Slice:output:0:movieId_embedding/movieId_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<movieId_embedding/movieId_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9movieId_embedding/movieId_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
4movieId_embedding/movieId_embedding_weights/GatherV2GatherV26movieId_embedding/to_sparse_input/dense_shape:output:0EmovieId_embedding/movieId_embedding_weights/GatherV2/indices:output:0BmovieId_embedding/movieId_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: т
2movieId_embedding/movieId_embedding_weights/Cast/xPack9movieId_embedding/movieId_embedding_weights/Prod:output:0=movieId_embedding/movieId_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:б
9movieId_embedding/movieId_embedding_weights/SparseReshapeSparseReshape1movieId_embedding/to_sparse_input/indices:index:06movieId_embedding/to_sparse_input/dense_shape:output:0;movieId_embedding/movieId_embedding_weights/Cast/x:output:0*-
_output_shapes
:         :Ъ
BmovieId_embedding/movieId_embedding_weights/SparseReshape/IdentityIdentitymovieId_embedding/values:y:0*
T0	*#
_output_shapes
:         |
:movieId_embedding/movieId_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R И
8movieId_embedding/movieId_embedding_weights/GreaterEqualGreaterEqualKmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0CmovieId_embedding/movieId_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:         б
1movieId_embedding/movieId_embedding_weights/WhereWhere<movieId_embedding/movieId_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:         М
9movieId_embedding/movieId_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ы
3movieId_embedding/movieId_embedding_weights/ReshapeReshape9movieId_embedding/movieId_embedding_weights/Where:index:0BmovieId_embedding/movieId_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
6movieId_embedding/movieId_embedding_weights/GatherV2_1GatherV2JmovieId_embedding/movieId_embedding_weights/SparseReshape:output_indices:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:         }
;movieId_embedding/movieId_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
6movieId_embedding/movieId_embedding_weights/GatherV2_2GatherV2KmovieId_embedding/movieId_embedding_weights/SparseReshape/Identity:output:0<movieId_embedding/movieId_embedding_weights/Reshape:output:0DmovieId_embedding/movieId_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         п
4movieId_embedding/movieId_embedding_weights/IdentityIdentityHmovieId_embedding/movieId_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:З
EmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ┌
SmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?movieId_embedding/movieId_embedding_weights/GatherV2_1:output:0?movieId_embedding/movieId_embedding_weights/GatherV2_2:output:0=movieId_embedding/movieId_embedding_weights/Identity:output:0NmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:         :         :         :         и
WmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       к
YmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
QmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЎ
JmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/UniqueUniquecmovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:         :         ╔
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather[movieid_embedding_movieid_embedding_weights_embedding_lookup_sparse_embedding_lookup_303391NmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:y:0*
Tindices0	*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303391*'
_output_shapes
:         
*
dtype0ъ
]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*n
_classd
b`loc:@movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/303391*'
_output_shapes
:         
Е
_movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityfmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         
╡
CmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparseSparseSegmentMeanhmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0PmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/Unique:idx:0ZmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:         
М
;movieId_embedding/movieId_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       г
5movieId_embedding/movieId_embedding_weights/Reshape_1ReshapeimovieId_embedding/movieId_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0DmovieId_embedding/movieId_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:         н
1movieId_embedding/movieId_embedding_weights/ShapeShapeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:Й
?movieId_embedding/movieId_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
AmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9movieId_embedding/movieId_embedding_weights/strided_sliceStridedSlice:movieId_embedding/movieId_embedding_weights/Shape:output:0HmovieId_embedding/movieId_embedding_weights/strided_slice/stack:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_1:output:0JmovieId_embedding/movieId_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3movieId_embedding/movieId_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :щ
1movieId_embedding/movieId_embedding_weights/stackPack<movieId_embedding/movieId_embedding_weights/stack/0:output:0BmovieId_embedding/movieId_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:ц
0movieId_embedding/movieId_embedding_weights/TileTile>movieId_embedding/movieId_embedding_weights/Reshape_1:output:0:movieId_embedding/movieId_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:         
├
6movieId_embedding/movieId_embedding_weights/zeros_like	ZerosLikeLmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
м
+movieId_embedding/movieId_embedding_weightsSelect9movieId_embedding/movieId_embedding_weights/Tile:output:0:movieId_embedding/movieId_embedding_weights/zeros_like:y:0LmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:         
ж
2movieId_embedding/movieId_embedding_weights/Cast_1Cast6movieId_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: В
8movieId_embedding/movieId_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:н
3movieId_embedding/movieId_embedding_weights/Slice_1Slice6movieId_embedding/movieId_embedding_weights/Cast_1:y:0BmovieId_embedding/movieId_embedding_weights/Slice_1/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:Ч
3movieId_embedding/movieId_embedding_weights/Shape_1Shape4movieId_embedding/movieId_embedding_weights:output:0*
T0*
_output_shapes
:Г
9movieId_embedding/movieId_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:Л
8movieId_embedding/movieId_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         │
3movieId_embedding/movieId_embedding_weights/Slice_2Slice<movieId_embedding/movieId_embedding_weights/Shape_1:output:0BmovieId_embedding/movieId_embedding_weights/Slice_2/begin:output:0AmovieId_embedding/movieId_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7movieId_embedding/movieId_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : к
2movieId_embedding/movieId_embedding_weights/concatConcatV2<movieId_embedding/movieId_embedding_weights/Slice_1:output:0<movieId_embedding/movieId_embedding_weights/Slice_2:output:0@movieId_embedding/movieId_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:х
5movieId_embedding/movieId_embedding_weights/Reshape_2Reshape4movieId_embedding/movieId_embedding_weights:output:0;movieId_embedding/movieId_embedding_weights/concat:output:0*
T0*'
_output_shapes
:         
Е
movieId_embedding/ShapeShape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%movieId_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'movieId_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'movieId_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
movieId_embedding/strided_sliceStridedSlice movieId_embedding/Shape:output:0.movieId_embedding/strided_slice/stack:output:00movieId_embedding/strided_slice/stack_1:output:00movieId_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!movieId_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
л
movieId_embedding/Reshape/shapePack(movieId_embedding/strided_slice:output:0*movieId_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:└
movieId_embedding/ReshapeReshape>movieId_embedding/movieId_embedding_weights/Reshape_2:output:0(movieId_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:         
\
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         o
concat/concatIdentity"movieId_embedding/Reshape:output:0*
T0*'
_output_shapes
:         
e
IdentityIdentityconcat/concat:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOpU^movieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2м
TmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookupTmovieId_embedding/movieId_embedding_weights/embedding_lookup_sparse/embedding_lookup:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
Ъ

Ї
C__inference_dense_1_layer_call_and_return_conditional_losses_303670

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
ыж
Ё
"__inference__traced_restore_303970
file_prefixV
Cassignvariableop_dense_features_movieid_embedding_embedding_weights:	щ
Z
Fassignvariableop_1_dense_features_1_userid_embedding_embedding_weights:
▒ъ
1
assignvariableop_2_dense_kernel:
+
assignvariableop_3_dense_bias:
3
!assignvariableop_4_dense_1_kernel:

-
assignvariableop_5_dense_1_bias:
3
!assignvariableop_6_dense_2_kernel:
-
assignvariableop_7_dense_2_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: 3
$assignvariableop_17_true_positives_1:	╚3
$assignvariableop_18_true_negatives_1:	╚4
%assignvariableop_19_false_positives_1:	╚4
%assignvariableop_20_false_negatives_1:	╚1
"assignvariableop_21_true_positives:	╚1
"assignvariableop_22_true_negatives:	╚2
#assignvariableop_23_false_positives:	╚2
#assignvariableop_24_false_negatives:	╚`
Massignvariableop_25_adam_dense_features_movieid_embedding_embedding_weights_m:	щ
b
Nassignvariableop_26_adam_dense_features_1_userid_embedding_embedding_weights_m:
▒ъ
9
'assignvariableop_27_adam_dense_kernel_m:
3
%assignvariableop_28_adam_dense_bias_m:
;
)assignvariableop_29_adam_dense_1_kernel_m:

5
'assignvariableop_30_adam_dense_1_bias_m:
;
)assignvariableop_31_adam_dense_2_kernel_m:
5
'assignvariableop_32_adam_dense_2_bias_m:`
Massignvariableop_33_adam_dense_features_movieid_embedding_embedding_weights_v:	щ
b
Nassignvariableop_34_adam_dense_features_1_userid_embedding_embedding_weights_v:
▒ъ
9
'assignvariableop_35_adam_dense_kernel_v:
3
%assignvariableop_36_adam_dense_bias_v:
;
)assignvariableop_37_adam_dense_1_kernel_v:

5
'assignvariableop_38_adam_dense_1_bias_v:
;
)assignvariableop_39_adam_dense_2_kernel_v:
5
'assignvariableop_40_adam_dense_2_bias_v:
identity_42ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9х
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*Л
valueБB■*BTlayer_with_weights-0/movieId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/userId_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/movieId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBolayer_with_weights-1/userId_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH─
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B є
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╛
_output_shapesл
и::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOpAssignVariableOpCassignvariableop_dense_features_movieid_embedding_embedding_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_1AssignVariableOpFassignvariableop_1_dense_features_1_userid_embedding_embedding_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_18AssignVariableOp$assignvariableop_18_true_negatives_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_19AssignVariableOp%assignvariableop_19_false_positives_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_20AssignVariableOp%assignvariableop_20_false_negatives_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_21AssignVariableOp"assignvariableop_21_true_positivesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_negativesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_25AssignVariableOpMassignvariableop_25_adam_dense_features_movieid_embedding_embedding_weights_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_26AssignVariableOpNassignvariableop_26_adam_dense_features_1_userid_embedding_embedding_weights_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_33AssignVariableOpMassignvariableop_33_adam_dense_features_movieid_embedding_embedding_weights_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_34AssignVariableOpNassignvariableop_34_adam_dense_features_1_userid_embedding_embedding_weights_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╒
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ┬
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ш

Є
A__inference_dense_layer_call_and_return_conditional_losses_303650

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
└
Х
(__inference_dense_1_layer_call_fn_303659

inputs
unknown:


	unknown_0:

identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302364o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╒

╫
&__inference_model_layer_call_fn_302847
inputs_movieid
inputs_userid
unknown:	щ

	unknown_0:
▒ъ

	unknown_1:

	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinputs_movieidinputs_useridunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_302388o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:         
(
_user_specified_nameinputs/movieId:RN
#
_output_shapes
:         
'
_user_specified_nameinputs/userId
╬
ж
1__inference_dense_features_1_layer_call_fn_303439
features_movieid
features_userid
unknown:
▒ъ

identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallfeatures_movieidfeatures_useridunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_302323o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:         
*
_user_specified_namefeatures/movieId:TP
#
_output_shapes
:         
)
_user_specified_namefeatures/userId
Ъ

Ї
C__inference_dense_1_layer_call_and_return_conditional_losses_302364

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▌
serving_default╔
7
movieId,
serving_default_movieId:0         
5
userId+
serving_default_userId:0         ;
dense_20
StatefulPartitionedCall:0         tensorflow/serving/predict:ю╦
═
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources
'#movieId_embedding/embedding_weights"
_tf_keras_layer
є
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_feature_columns
"
_resources
&#"userId_embedding/embedding_weights"
_tf_keras_layer
е
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias"
_tf_keras_layer
╗
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
╗
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
X
0
#1
02
13
84
95
@6
A7"
trackable_list_wrapper
X
0
#1
02
13
84
95
@6
A7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
═
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32т
&__inference_model_layer_call_fn_302407
&__inference_model_layer_call_fn_302847
&__inference_model_layer_call_fn_302869
&__inference_model_layer_call_fn_302741┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
╣
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32╬
A__inference_model_layer_call_and_return_conditional_losses_303057
A__inference_model_layer_call_and_return_conditional_losses_303245
A__inference_model_layer_call_and_return_conditional_losses_302768
A__inference_model_layer_call_and_return_conditional_losses_302795┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
╘B╤
!__inference__wrapped_model_302140movieIduserId"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemЬ#mЭ0mЮ1mЯ8mа9mб@mвAmгvд#vе0vж1vз8vи9vй@vкAvл"
	optimizer
,
Tserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
Ztrace_0
[trace_12╕
/__inference_dense_features_layer_call_fn_303253
/__inference_dense_features_layer_call_fn_303261╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zZtrace_0z[trace_1
е
\trace_0
]trace_12ю
J__inference_dense_features_layer_call_and_return_conditional_losses_303346
J__inference_dense_features_layer_call_and_return_conditional_losses_303431╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z\trace_0z]trace_1
 "
trackable_list_wrapper
"
_generic_user_object
E:C	щ
22dense_features/movieId_embedding/embedding_weights
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
н
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
є
ctrace_0
dtrace_12╝
1__inference_dense_features_1_layer_call_fn_303439
1__inference_dense_features_1_layer_call_fn_303447╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zctrace_0zdtrace_1
й
etrace_0
ftrace_12Є
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303532
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303617╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zetrace_0zftrace_1
 "
trackable_list_wrapper
"
_generic_user_object
G:E
▒ъ
23dense_features_1/userId_embedding/embedding_weights
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ё
ltrace_02╙
,__inference_concatenate_layer_call_fn_303623в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zltrace_0
Л
mtrace_02ю
G__inference_concatenate_layer_call_and_return_conditional_losses_303630в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zmtrace_0
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
н
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ъ
strace_02═
&__inference_dense_layer_call_fn_303639в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zstrace_0
Е
ttrace_02ш
A__inference_dense_layer_call_and_return_conditional_losses_303650в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zttrace_0
:
2dense/kernel
:
2
dense/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
н
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ь
ztrace_02╧
(__inference_dense_1_layer_call_fn_303659в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zztrace_0
З
{trace_02ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_303670в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z{trace_0
 :

2dense_1/kernel
:
2dense_1/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
о
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ю
Бtrace_02╧
(__inference_dense_2_layer_call_fn_303679в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0
Й
Вtrace_02ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_303690в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
 :
2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
@
Г0
Д1
Е2
Ж3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
&__inference_model_layer_call_fn_302407movieIduserId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
&__inference_model_layer_call_fn_302847inputs/movieIdinputs/userId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
&__inference_model_layer_call_fn_302869inputs/movieIdinputs/userId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
&__inference_model_layer_call_fn_302741movieIduserId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
йBж
A__inference_model_layer_call_and_return_conditional_losses_303057inputs/movieIdinputs/userId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
йBж
A__inference_model_layer_call_and_return_conditional_losses_303245inputs/movieIdinputs/userId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
A__inference_model_layer_call_and_return_conditional_losses_302768movieIduserId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
A__inference_model_layer_call_and_return_conditional_losses_302795movieIduserId"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╤B╬
$__inference_signature_wrapper_302825movieIduserId"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
/__inference_dense_features_layer_call_fn_303253features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
пBм
/__inference_dense_features_layer_call_fn_303261features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩B╟
J__inference_dense_features_layer_call_and_return_conditional_losses_303346features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩B╟
J__inference_dense_features_layer_call_and_return_conditional_losses_303431features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▒Bо
1__inference_dense_features_1_layer_call_fn_303439features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒Bо
1__inference_dense_features_1_layer_call_fn_303447features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠B╔
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303532features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠B╔
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303617features/movieIdfeatures/userId"╙
╩▓╞
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
,__inference_concatenate_layer_call_fn_303623inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
G__inference_concatenate_layer_call_and_return_conditional_losses_303630inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_dense_layer_call_fn_303639inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_layer_call_and_return_conditional_losses_303650inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_dense_1_layer_call_fn_303659inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_dense_1_layer_call_and_return_conditional_losses_303670inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_dense_2_layer_call_fn_303679inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_dense_2_layer_call_and_return_conditional_losses_303690inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
З	variables
И	keras_api

Йtotal

Кcount"
_tf_keras_metric
c
Л	variables
М	keras_api

Нtotal

Оcount
П
_fn_kwargs"
_tf_keras_metric
Р
Р	variables
С	keras_api
Тtrue_positives
Уtrue_negatives
Фfalse_positives
Хfalse_negatives"
_tf_keras_metric
Р
Ц	variables
Ч	keras_api
Шtrue_positives
Щtrue_negatives
Ъfalse_positives
Ыfalse_negatives"
_tf_keras_metric
0
Й0
К1"
trackable_list_wrapper
.
З	variables"
_generic_user_object
:  (2total
:  (2count
0
Н0
О1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
@
Т0
У1
Ф2
Х3"
trackable_list_wrapper
.
Р	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
@
Ш0
Щ1
Ъ2
Ы3"
trackable_list_wrapper
.
Ц	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
J:H	щ
29Adam/dense_features/movieId_embedding/embedding_weights/m
L:J
▒ъ
2:Adam/dense_features_1/userId_embedding/embedding_weights/m
#:!
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
%:#

2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
%:#
2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
J:H	щ
29Adam/dense_features/movieId_embedding/embedding_weights/v
L:J
▒ъ
2:Adam/dense_features_1/userId_embedding/embedding_weights/v
#:!
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
%:#

2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
%:#
2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v╔
!__inference__wrapped_model_302140г#0189@Adвa
ZвW
UкR
(
movieIdК
movieId         
&
userIdК
userId         
к "1к.
,
dense_2!К
dense_2         ╧
G__inference_concatenate_layer_call_and_return_conditional_losses_303630ГZвW
PвM
KЪH
"К
inputs/0         

"К
inputs/1         

к "%в"
К
0         
Ъ ж
,__inference_concatenate_layer_call_fn_303623vZвW
PвM
KЪH
"К
inputs/0         

"К
inputs/1         

к "К         г
C__inference_dense_1_layer_call_and_return_conditional_losses_303670\89/в,
%в"
 К
inputs         

к "%в"
К
0         

Ъ {
(__inference_dense_1_layer_call_fn_303659O89/в,
%в"
 К
inputs         

к "К         
г
C__inference_dense_2_layer_call_and_return_conditional_losses_303690\@A/в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ {
(__inference_dense_2_layer_call_fn_303679O@A/в,
%в"
 К
inputs         

к "К         √
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303532к#~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p 
к "%в"
К
0         

Ъ √
L__inference_dense_features_1_layer_call_and_return_conditional_losses_303617к#~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p
к "%в"
К
0         

Ъ ╙
1__inference_dense_features_1_layer_call_fn_303439Э#~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p 
к "К         
╙
1__inference_dense_features_1_layer_call_fn_303447Э#~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p
к "К         
∙
J__inference_dense_features_layer_call_and_return_conditional_losses_303346к~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p 
к "%в"
К
0         

Ъ ∙
J__inference_dense_features_layer_call_and_return_conditional_losses_303431к~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p
к "%в"
К
0         

Ъ ╤
/__inference_dense_features_layer_call_fn_303253Э~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p 
к "К         
╤
/__inference_dense_features_layer_call_fn_303261Э~в{
tвq
gкd
1
movieId&К#
features/movieId         
/
userId%К"
features/userId         

 
p
к "К         
б
A__inference_dense_layer_call_and_return_conditional_losses_303650\01/в,
%в"
 К
inputs         
к "%в"
К
0         

Ъ y
&__inference_dense_layer_call_fn_303639O01/в,
%в"
 К
inputs         
к "К         
х
A__inference_model_layer_call_and_return_conditional_losses_302768Я#0189@Alвi
bв_
UкR
(
movieIdК
movieId         
&
userIdК
userId         
p 

 
к "%в"
К
0         
Ъ х
A__inference_model_layer_call_and_return_conditional_losses_302795Я#0189@Alвi
bв_
UкR
(
movieIdК
movieId         
&
userIdК
userId         
p

 
к "%в"
К
0         
Ъ є
A__inference_model_layer_call_and_return_conditional_losses_303057н#0189@Azвw
pвm
cк`
/
movieId$К!
inputs/movieId         
-
userId#К 
inputs/userId         
p 

 
к "%в"
К
0         
Ъ є
A__inference_model_layer_call_and_return_conditional_losses_303245н#0189@Azвw
pвm
cк`
/
movieId$К!
inputs/movieId         
-
userId#К 
inputs/userId         
p

 
к "%в"
К
0         
Ъ ╜
&__inference_model_layer_call_fn_302407Т#0189@Alвi
bв_
UкR
(
movieIdК
movieId         
&
userIdК
userId         
p 

 
к "К         ╜
&__inference_model_layer_call_fn_302741Т#0189@Alвi
bв_
UкR
(
movieIdК
movieId         
&
userIdК
userId         
p

 
к "К         ╦
&__inference_model_layer_call_fn_302847а#0189@Azвw
pвm
cк`
/
movieId$К!
inputs/movieId         
-
userId#К 
inputs/userId         
p 

 
к "К         ╦
&__inference_model_layer_call_fn_302869а#0189@Azвw
pвm
cк`
/
movieId$К!
inputs/movieId         
-
userId#К 
inputs/userId         
p

 
к "К         ╟
$__inference_signature_wrapper_302825Ю#0189@A_в\
в 
UкR
(
movieIdК
movieId         
&
userIdК
userId         "1к.
,
dense_2!К
dense_2         