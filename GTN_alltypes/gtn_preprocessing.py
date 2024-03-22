# -*- coding: utf-8 -*-
"""GTN_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fb5XBg4qX2J-jc6grQeXnPhXOnGh9d94
"""

import json
import numpy as np
from scipy.sparse import csr_matrix
import os
import subprocess
import sys
from json_tricks import loads
import glob
from fgtn_main import trainTheModel

standard_size=1000
batch_size=6

standard_sizexbatch_size=8000

class MyClass:
    pass

def getobj(file):
   while True:
    s=file.read(1)
    if not s:
        return s
    if s=='{':
       break
   depth=1
   while depth>0:
      char=file.read(1)
      if char=='{':
         depth+=1
      if char=='}':
         depth-=1
      s+=char
   return s

primitive_types=["void", "byte", "short", "int", "long", "float", "double", "char", "boolean"]
nodeList=["MethodDeclaration", "Parameter", "FieldDeclaration", "ArrayType", "ClassOrInterfaceType", "VariableDeclarationExpr"]
types=['CompilationUnit', 'PackageDeclaration', 'Name', 'ImportDeclaration', 'ClassOrInterfaceDeclaration', 'Modifier', 'SimpleName', 'FieldDeclaration', 'intModifier', 'public Modifier', 'static Modifier', 'final Modifier', 'VariableDeclarator', 'PrimitiveType', 'UnaryExpr', 'IntegerLiteralExpr', 'ClassOrInterfaceType', 'StringLiteralExpr', 'AnnotationDeclaration', 'NormalAnnotationExpr', 'MemberValuePair', 'ArrayInitializerExpr', 'NameExpr', 'private Modifier', 'booleanModifier', 'ConstructorDeclaration', 'Parameter', 'MarkerAnnotationExpr', 'BackTargetTypeMarker', 'BlockStmt', 'ExpressionStmt', 'AssignExpr', 'NonNullMarker', 'MethodCallExpr', 'FieldAccessExpr', 'MethodDeclaration', 'OverrideMarker', 'VoidType', 'ReturnStmt', 'IfStmt', 'BinaryExpr', 'NullLiteralExpr', 'VariableDeclarationExpr', 'ObjectCreationExpr', 'ArrayType', 'ArrayCreationExpr', 'ArrayCreationLevel', 'CharLiteralExpr', 'SwitchStmt', 'SwitchEntry', 'SingleMemberAnnotationExpr', 'ThisExpr', 'SystemApiMarker', 'abstract Modifier', 'DeprecatedMarker', 'ConfigFlagMarker', 'BooleanLiteralExpr', 'ThrowStmt', 'byteModifier', 'shortModifier', 'ForEachStmt', 'UnsupportedAppUsageMarker', 'floatModifier', 'TryStmt', 'ConditionalExpr', 'ExplicitConstructorInvocationStmt', 'CastExpr', 'Vr2dDisplayFlagMarker', 'EnclosedExpr', 'longModifier', 'BreakStmt', 'BlockComment', 'InstanceOfExpr', 'ProfileTypeMarker', 'CallbackExecutorMarker', 'LambdaExpr', 'charModifier', 'ForStmt', 'ClassExpr', 'doubleModifier', 'ArrayAccessExpr', 'WhileStmt', 'SuperExpr', 'AttributionFlagsMarker', 'default Modifier', 'ChangeIdMarker', 'FlagsMarker', 'InputConfigFlagsMarker', 'SynchronizedStmt', 'LineComment', 'TypeParameter', 'WildcardType', 'UserIdIntMarker', 'UnknownType', 'SysUISingletonMarker', 'InjectMarker', 'ContinueStmt', 'synchronized Modifier', 'native Modifier', 'protected Modifier', 'TypeMarker', 'CallSuperMarker', 'NotifyEventTypeMarker', 'InitializerDeclaration', 'ResultMarker', 'volatile Modifier', 'VisibleForTestingMarker', 'LongLiteralExpr', 'TestApiMarker', 'IpcDataCacheModuleMarker', 'WorkerThreadMarker', 'JavadocComment', 'NetworkTypeMarker', 'CurrentTimeMillisLongMarker', 'ColorIntMarker', 'ColorLongMarker', 'DoubleLiteralExpr', 'GameModeMarker', 'UserHandleAwareMarker', 'MethodReferenceExpr', 'TypeExpr', 'RenderModeMarker', 'PermissionTypeMarker', 'MemberMarker', 'AnyThreadMarker', 'InputMethodNavButtonFlagsMarker', 'BreakStrategyMarker', 'HyphenationFrequencyMarker', 'JustificationModeMarker', 'PxMarker', 'FastNativeMarker', 'CriticalNativeMarker', 'OriginEnumMarker', 'PurposeEnumMarker', 'EncryptionPaddingEnumMarker', 'SignaturePaddingEnumMarker', 'DigestEnumMarker', 'BlockModeEnumMarker', 'AuthEnumMarker', 'SecurityLevelEnumMarker', 'MainMarker', 'LetterboxHorizontalReachabilityPositionMarker', 'LetterboxVerticalReachabilityPositionMarker', 'OriginMarker', 'InsetsTypeMarker', 'DispatchModeMarker', 'SecurityPatchStateMarker', 'LevelMarker', 'KeyAlgorithmEnumMarker', 'StateMarker', 'AutofillTypeMarker', 'RotationMarker', 'VibrationIntensityMarker', 'StringResMarker', 'AttrResMarker', 'BytesLongMarker', 'PartitionTypeMarker', 'AppearanceMarker', 'ActionTypeMarker', 'FlagTypeMarker', 'RequestFlagsMarker', 'AnimationTypeMarker', 'transient Modifier', 'UiThreadMarker', 'AssertStmt', 'ActivityTypeMarker', 'AvailabilityMarker', 'RequestTemplateMarker', 'ErrorCodeMarker', 'CAMERA_AUDIO_RESTRICTIONMarker', 'CapabilityStateMarker', 'MainThreadMarker', 'ImmutableMarker', 'SamplingStrategyMarker', 'EnumDeclaration', 'EnumConstantDeclaration', 'KeyguardBouncerScopeMarker', 'LockoutModeMarker', 'DrawableResMarker', 'IconTypeMarker', 'ChangeTypeMarker', 'SettingMarker', 'TransitionOldTypeMarker', 'RemoteViewMarker', 'StyleResMarker', 'RemotableViewMethodMarker', 'RecognitionFlagsMarker', 'ConfigMarker', 'ImplementationMarker', 'DirectMarker', 'StatusMarker', 'RuleMarker', 'BeforeMarker', 'AfterMarker', 'TestMarker', 'ViewportTypeMarker', 'EnrollReasonMarker', 'SensorTypeMarker', 'ElapsedRealtimeLongMarker', 'EmptyStmt', 'StaticMarker', 'WindowingModeMarker', 'PriorityMarker', 'ConnectorMarker', 'PermissionInfoFlagsMarker', 'PermissionWhitelistFlagsMarker', 'ProvidesMarker', 'CentralSurfacesScopeMarker', 'BindsMarker', 'IntoSetMarker', 'ConnectionToSinkTypeMarker', 'QueryFlagsMarker', 'EventTypesFlagMarker', 'CiphersuiteMarker', 'TransitionTypeMarker', 'TranslationFlagMarker', 'ApplyStrategyMarker', 'OrientationMarker', 'RequestTypeMarker', 'EventTypeMarker', 'ReadModeMarker', 'TransitionDirectionMarker', 'DoStmt', 'DocumentedMarker', 'DurationMillisLongMarker', 'SilentHeaderMarker', 'DismissalSurfaceMarker', 'DismissalSentimentMarker', 'ResponseResultMarker', 'DataFormatMarker', 'WriteModeMarker', 'StartResultMarker', 'StartArgFlagsMarker', 'StopForegroundFlagsMarker', 'EventMarker', 'AuthorizationStateMarker', 'ForegroundServiceTypeMarker', 'WakeReasonMarker', 'GoToSleepReasonMarker', 'ResultCodeMarker', 'PresubmitMarker', 'SmallTestMarker', 'BinderThreadMarker', 'TemplateTypeMarker', 'FormatMarker', 'LargeTestMarker', 'UiThreadTestMarker', 'ResponseCodeMarker', 'SessionModeMarker', 'SendRequestMarker', 'SendResultMarker', 'UiTemplateTypeMarker', 'CardStateInfoMarker', 'CheckResultMarker', 'ShortcutTypeMarker', 'AccessibilityFragmentTypeMarker', 'CinematicEffectStatusCodeMarker', 'ImageContentTypeMarker', 'StandbyBucketsMarker', 'ForcedReasonsMarker', 'ProcessStateMarker', 'AppActionMarker', 'AttestationProfileIdMarker', 'ViewModeMarker', 'ServiceStatusMarker', 'WarningTypeMarker', 'LayoutlibDelegateMarker', 'MissingMethodFlagsMarker', '/*package*/\nlongModifier', 'SearchIndexableMarker', 'MockMarker', 'CheckForNullMarker', 'NullableDeclMarker', 'NullableTypeMarker', 'NullAllowedMarker', 'NullUnknownMarker', 'NonnullMarker', 'NotNullMarker', 'NonNullDeclMarker', 'NonNullTypeMarker', 'NonNullByDefaultMarker', 'ParametersAreNonnullByDefaultMarker', 'ArrayResMarker', 'IgnoreMarker', 'StabilityMarker', 'PreciseCallStatesMarker', 'NrVopsStatusMarker', 'NrEmcStatusMarker', 'ImsStateMarker', 'ImsServiceCapabilityMarker', 'TransportTypeMarker', 'ExternalCallStateMarker', 'ImsRegistrationTechMarker', 'FeatureTypeMarker', 'PositionMarker', 'AppTypeMarker', 'IntRangeMarker', 'BiopMessageTypeMarker', 'LayoutResMarker', 'ThrottlingStatusMarker', 'KeyEventActionMarker', 'GwpAsanModeMarker', 'MemtagModeMarker', 'NativeHeapZeroInitializedMarker', 'StagedOperationTypeMarker', 'DistroStatusMarker', 'NotifyFlagsMarker', 'MotionEventActionMarker', 'ExtconDeviceTypeMarker', 'CallAudioRouteMarker', 'DeviceConfigKeyMarker', 'EventCategoryMarker', 'LetterboxBackgroundTypeMarker', 'LetterboxReachabilityPositionMarker', 'VariantMarker', 'ViewTypeMarker', 'FunctionalInterfaceMarker', 'RepeatModeMarker', 'BackgroundMarker', 'DimensionMarker', 'ColorResMarker', 'RawResMarker', 'ReactMethodMarker', 'GravityFlagMarker', 'LocalClassDeclarationStmt', 'AnimResMarker', 'GetterMarker', 'Slf4jMarker', 'SetterMarker', 'IdResMarker', 'screenTypeMarker', 'AutoAccessMarker', 'LightnessMarker', 'CalledByNativeMarker', 'AnnotationMemberDeclaration', 'NotThreadSafeMarker', 'ThreadSafeMarker', 'ErrorTypeMarker', 'JavascriptInterfaceMarker', 'SigninAccessPointMarker', 'ModalDialogTypeMarker', 'SearchEnginePromoTypeMarker', 'ReauthScopeMarker', 'OverrideStateMarker', 'SafeVarargsMarker', 'MediumTestMarker', 'EncodingMarker', 'TextSizeTypeMarker', 'ResizeModeMarker', 'StreamTypeMarker', 'StereoModeMarker', 'KeepMarker', 'BeforeClassMarker', 'FlashModeMarker', 'SubscribeMarker', 'MenuResMarker', 'AnimatorResMarker', 'AutoRestoreMarker', 'SingletonMarker', 'StatusCodeMarker', 'ActivityScopeMarker', 'LabeledStmt', 'KeycodeMarker', 'DraggableItemStateFlagsMarker', 'ScrollDirectionMarker', 'DimenResMarker', 'InternalApiMarker', 'JsonCreatorMarker', 'JsonIgnoreMarker', 'JsonPropertyMarker', 'DoNotStripMarker', 'UIManagerTypeMarker', 'ImageEventTypeMarker', 'ModeMarker', 'WMSingletonMarker', 'ShellMainThreadMarker', 'PropMarker', 'LinearColorMarker', 'EntityMarker', 'EntityInstanceMarker', 'CarProtocolMarker', 'KeepFieldsMarker', 'CarIconTypeMarker', 'ExperimentalCarApiMarker', 'CarColorTypeMarker', 'DoNotInlineMarker', 'AutoValueMarker', 'CarZoneRowMarker', 'CarZoneColumnMarker', 'StyleableResMarker', 'NonParcelFieldMarker', 'InputModeMarker', 'ReplaceStrategyMarker', 'ImageModeMarker', 'EvConnectorTypeMarker', 'FuelTypeMarker', 'NestedScrollTypeMarker', 'HorizontalAlignmentMarker', 'CommandVersionMarker', 'BuilderMarker', 'CanIgnoreReturnValueMarker', 'ProtoLayoutExperimentalMarker', 'SplitFinishBehaviorMarker', 'ExtLayoutDirectionMarker', 'RatioMarker', 'ComplicationTypeMarker', 'SplitPlaceholderFinishBehaviorMarker', 'AppCompatShadowedAttributesMarker', 'LongPropertyMarker', 'StringPropertyMarker', 'IdMarker', 'HvacFanDirectionMarker', 'AccessErrorMarker', 'VarType', 'TabSelectionTypeMarker', 'NativeMethodsMarker', 'InitializerMarker', 'SectionTypeMarker', 'GETMarker', 'UrlMarker', 'BodyMarker', 'ValueMarker', 'VideoProjectionFlagsMarker', 'SyntheticMarker', 'PureMarker', 'LogLevelMarker', 'PlaybackSuppressionReasonMarker', 'MessageTypeMarker', 'MonotonicNonNullMarker', 'TrackTypeMarker', 'CapabilitiesMarker', 'ColorSpaceMarker', 'ColorRangeMarker', 'ColorTransferMarker', 'VolumeFlagsMarker', 'VideoOutputModeMarker', 'RequirementFlagsMarker', 'AudioContentTypeMarker', 'AudioFlagsMarker', 'AudioUsageMarker', 'AudioAllowedCapturePolicyMarker', 'SpatializationBehaviorMarker', 'PcmEncodingMarker', 'TabCreationStateMarker', 'SecureModeMarker', 'TabLaunchTypeMarker', 'BufferReplacementModeMarker', 'BindsOptionalOfMarker', 'OptionalBooleanMarker', 'FontSizeUnitMarker', 'ExperimentalMarker', 'PropDefaultMarker', 'DirtinessStateMarker', 'AdaptiveToolbarButtonVariantMarker', 'StateChangeReasonMarker', 'ExpoMethodMarker', 'HiddenApiMarker', 'SlowMarker', 'ServiceMarker', 'IncubatingMarker', 'BetaMarker', 'TaskActionMarker', 'InputMarker', 'OptionalMarker', 'OutputDirectoryMarker', 'InputFilesMarker', 'NestedMarker', 'NonNlsMarker', 'InternalMarker', 'ModifierConstantMarker', 'TestOnlyMarker', 'XmlTransientMarker', 'InputFileMarker', 'ClassRuleMarker', 'AfterClassMarker', 'ParameterMarker', 'MemoizedMarker', 'AssistedMarker', 'GerritServerConfigMarker', 'SendEmailExecutorMarker', 'GerritPersonIdentMarker', 'AssistedInjectMarker', 'UiFieldMarker', 'MorphiaInternalMarker', 'SubstituteMarker', 'AliasMarker', 'ParameterizedTestMarker', 'BeforeEachMarker', 'AfterEachMarker', 'NewFieldMarker', 'TraceMarker', 'FieldNameMarker', 'DataMarker', 'ComponentMarker', 'RequiredArgsConstructorMarker', 'ToStringMarker', 'XmlElementMarker', 'ColumnMarker', 'ValidMarker', 'ManagedDataMarker', 'ManagedAttributeMarker', 'NoArgsConstructorMarker', 'AllArgsConstructorMarker', 'ControllerMarker', 'AutowiredMarker', 'ExtensionMarker', 'DataBoundConstructorMarker', 'RequirePOSTMarker', 'ExportedBeanMarker', 'DataBoundSetterMarker', 'QueryParameterMarker', 'PrivateMarker', 'TestExtensionMarker', 'CheckReturnValueMarker', 'AncestorInPathMarker', 'POSTMarker', 'UtilityClassMarker', 'WhitelistedMarker', 'PostConstructMarker', 'SneakyThrowsMarker', 'EqualsAndHashCodeMarker', 'TransientMarker', 'DefaultMarker', 'RequiredMarker', 'RestControllerMarker', 'NonNullApiMarker', 'NonNullFieldsMarker', 'ParamMarker', 'HeaderMarker', 'RequestParamMarker', 'ValidatedMarker', 'NonnegativeMarker', 'JsonpDeserializableMarker', 'EditableMarker', 'NotEmptyMarker', 'BeforeAllMarker', 'AfterAllMarker', 'EvolvingMarker', 'GwtCompatibleMarker', 'WeakMarker', 'GwtIncompatibleMarker', 'WeakOuterMarker', 'ViewComponentMarker', 'J2ObjCIncompatibleMarker', '/* static */\nlongModifier', 'ForOverrideMarker', 'DerivedMarker', 'CheckMarker', 'AddToRuleKeyMarker', 'BuckStyleValueMarker', 'OnChannelThreadMarker', 'OnClientThreadMarker', 'UnknownKeyForMarker', 'InitializedMarker', 'ProcessElementMarker', 'ElementMarker', 'WithBeanGetterMarker', 'JsonAutoDetectMarker', 'ObjectIdMarker', 'WithSpanMarker', 'ConfigurationMarker', 'NotBlankMarker', 'ContextMarker', 'TimedMarker', 'DELETEMarker', 'PositiveMarker', 'PositiveOrZeroMarker', 'AlphaMarker', 'AccessesPartialKeyMarker', 'AutoCodecMarker', 'InstantiatorMarker', 'VisibleForSerializationMarker', 'SerializationConstantMarker', 'StarlarkConstructorMarker', 'NamedMarker', 'PublicEvolvingMarker', 'RpcTimeoutMarker', 'BenchmarkMarker', 'NullFromTypeParamMarker', 'UnmodifiableMarker', 'ReferenceMarker', 'SerialMarker', 'ActivateMarker', 'DeactivateMarker', 'JaxrsResourceMarker', 'JSONRequiredMarker', 'WebSocketMarker', 'OnWebSocketCloseMarker', 'OnWebSocketConnectMarker', 'OnWebSocketMessageMarker', 'OnWebSocketErrorMarker', 'ModifiedMarker', 'ExposeMarker', 'PreDestroyMarker', 'EventHandlerMarker', 'NlsSafeMarker', 'NlsMarker', 'ExcludeMarker', 'ShadowMarker', 'TransactionalMarker', 'FinalDefaultMarker', 'ConcurrentMethodMarker', 'OverridingMethodsMustInvokeSuperMarker', 'DialogTitleMarker', 'Log4j2Marker', 'BeanMarker', 'ResourceMarker', 'TooltipMarker', 'DialogMessageMarker', 'ButtonMarker', 'StubbedMarker', 'NotificationTitleMarker', 'ProgressTitleMarker', 'ActionTextMarker', 'InspectionMessageMarker', 'NotificationContentMarker', 'IntentionFamilyNameMarker', 'IntentionNameMarker', 'SafeFieldForPreviewMarker', 'SupportMarker', 'JsNonNullMarker', 'NullMarkedMarker', 'KtPropertyMarker', 'AutoConfigurationMarker', 'JmixPropertyMarker', 'RequestBodyMarker', 'ReadOperationMarker', 'SelectorMarker', 'ParameterizedAdapterTestMarker', 'PathVariableMarker', 'GetExchangeMarker', 'TestValueMarker', 'EnableCachingMarker', 'ParameterizedHttpServerTestMarker', 'PrimaryMarker', 'ConditionalOnMissingBeanMarker', 'ProgrammaticMarker', 'SpringIntegrationTestMarker', 'CreatedDateMarker', 'LastModifiedDateMarker', 'EnableBatchProcessingMarker', 'RepositoryMarker', 'MemberSupportMarker', 'GraphQlExceptionHandlerMarker', 'VolatileMarker', 'CopyMarker', 'InitMarker', 'Other']

directory = '/home/k/ks225/annotations/spectral_2hv/'
files = glob.glob(os.path.join(directory, '*' + '.json'))

datadir='/home/k/ks225/nullproj/GTN_alltypes/data/Null/'

# Define the custom order
custom_order = ["data1.json", "data0.json", "data3.json", "data4.json", "data2.json"]
files_sorted = sorted(files, key=lambda x: custom_order.index(os.path.basename(x)))

for fname in files_sorted:
  print(fname)

  '''
  # Load JSON data from file
  with open(fname, 'r') as f:
    json_str = f.read()

  # Use json_tricks.loads to convert the JSON string to a Python object
  json_data = loads(json_str)
  '''

  json_data = []
  #types=set()

  # Load JSON data from file
  with open(fname, "r") as file:
      while True:
          obj_str = getobj(file)
          if not obj_str:
            break
          obj_str = loads(obj_str)
          json_data.append(obj_str)
          #nodes=obj_str['nodes']
          #for node in nodes:
            #types.add(node['type'])

  print(len(types))

  print(types)

  #json_data=list(np.random.choice(json_data,batch_size if len(json_data)>batch_size else len(json_data),replace=False))

  #print(len(json_data))
    
  if len(json_data)<2:
    print("no data")
    continue

  startidx=0
  emptyflag=False

  while not emptyflag:
    json_data_batch=[]
    needToAdd=batch_size

    while needToAdd>0 and startidx<len(json_data):
      if len(json_data[startidx]['nodes'])>500:
        json_data_batch.append(json_data[startidx])
        needToAdd-=1
      startidx+=1

    if not startidx<len(json_data):
      emptyflag=True

    print(len(json_data_batch))
      
    if len(json_data_batch)<1:
      print("no data")
      continue

    type(json_data_batch)

    #type(json_data_batch[0]['adjacencyList'])

    json_data_batch[0]

    #type(json_data_batch[0]['adjacencyList'][str(json_data_batch[0]['nodes'][350]['id'])])

    #types=list(types)
    #types.index("Name")

    """# Edges"""

    nnode=0
    nmax=0

    for graph_json in json_data_batch:
      nodes=graph_json['nodes']
      for node in nodes:
        nnode+=1
        nmax=max(nmax, node['id'])

    print('nnode = ',nnode)
    print('nmax = ',nmax)

    nterm=0

    for graph_json in json_data_batch:
      nterm+=len(graph_json['nameList'])

    print('nterm = ',nterm)

    nnum=nnode+nterm

    print("nnum = ",nnum)

    A_n=np.zeros((standard_sizexbatch_size, standard_sizexbatch_size))

    gnode=0
    csum=0

    for graph_json in json_data_batch:
      nodes=graph_json['nodes']
      alist=graph_json['adjacencyList']
      for node in nodes:
        if str(node['id']) in alist:
          for neighbor in alist[str(node['id'])]:
            A_n[gnode,neighbor+csum]=1
        gnode+=1
      csum+=len(nodes)

    A_n=csr_matrix(A_n)

    A_t=np.zeros((standard_sizexbatch_size, standard_sizexbatch_size))

    tsum=0
    csum=0

    for graph_json in json_data_batch:
      nodes=len(graph_json['nodes'])
      nlist=graph_json['nameList']
      tnode=0
      for tkey in nlist:
        for node in nlist[tkey]:
          A_t[csum+node,nnode+tsum+tnode]=1
        tnode+=1
      csum+=nodes
      tsum+=tnode

    A_t=csr_matrix(A_t)

    import pickle

    edges=[A_n,A_n.transpose(),A_t,A_t.transpose()]
    with open(datadir+'edges.pkl', 'wb') as f:
      pickle.dump(edges, f)

    """# Node Features"""

    #last feature is nullable
    null_feat=np.zeros((standard_sizexbatch_size,2))
    type_feat=np.zeros((standard_sizexbatch_size,len(types)+1))

    gnode=0

    snowflakes=[]

    for graph_json in json_data_batch:
      nodes=graph_json['nodes']
      for node in nodes:
        if any(x in nodeList for x in node['type']) and all(x not in primitive_types for x in node['type']):
          snowflakes.append(gnode)
        if node['nullable']==0:
          null_feat[gnode,0]=1
        else:
          null_feat[gnode,1]=1
        for ind_t in node['type']:
          if ind_t in types:
            type_feat[gnode,types.index(ind_t)]=1
        gnode+=1

    print("snowflakes =",len(snowflakes))

    for node_i in range(gnode,standard_sizexbatch_size):
      null_feat[node_i,0]=1
      type_feat[node_i,len(types)]=1

    #node_feature=np.concatenate((null_feat,type_feat), axis=1)
    with open(datadir+'node_features.pkl', 'wb') as f:
      pickle.dump(type_feat, f)

    """# Label"""

    nullnodes=0

    for graph_json in json_data_batch:
      nodes=graph_json['nodes']
      for node in nodes:
        if node['nullable']==1:
          nullnodes+=1

    nullnodes

    # Find the row indices where the value in the specified column is 1
    row_indices = [x for x in np.where(null_feat[:, 1] == 1)[0] if x in snowflakes]
    #row_indices = np.where(null_feat[:, 1] == 1)[0]

    print(len(row_indices))

    if not row_indices:
      continue

    train_size=int(0.6*len(row_indices))
    other_size=int(0.2*len(row_indices))
    print('Train = ', train_size)
    print('Valid/Test = ', other_size)

    nonnullable_idx = [x for x in np.where(null_feat[:, 0] == 1)[0] if x < nnode and x in snowflakes]
    #nonnullable_idx = [x for x in np.where(null_feat[:, 0] == 1)[0] if x < nnode]

    print(len(nonnullable_idx))

    if not nonnullable_idx:
      continue

    train_null=list(np.random.choice(row_indices,train_size,replace=False))
    train_nonnull=list(np.random.choice(nonnullable_idx,train_size,replace=False))

    train_idx=np.array(train_null + train_nonnull)
    #print(train_idx)
    if len(train_idx)<1:
      print("train_idx empty")
      continue
    train_target=null_feat[train_idx,1]

    train_label = np.vstack((train_idx,train_target)).transpose()

    remaining_null=[item for item in row_indices if item not in train_null]
    remaining_nonnull=[item for item in nonnullable_idx if item not in train_nonnull]

    valid_null=list(np.random.choice(remaining_null,other_size,replace=False))
    valid_nonnull=list(np.random.choice(remaining_nonnull,other_size,replace=False))

    valid_idx=np.array(valid_null + valid_nonnull)
    if len(valid_idx)<1:
      print("valid_idx empty")
      continue
    valid_target=null_feat[valid_idx,1]

    valid_label = np.vstack((valid_idx,valid_target)).transpose()

    remaining_null=[item for item in remaining_null if item not in valid_null]
    remaining_nonnull=[item for item in remaining_nonnull if item not in valid_nonnull]

    test_idx=np.array(remaining_null+remaining_nonnull)
    if len(test_idx)<1:
      print("test_idx empty")
      continue
    test_target=null_feat[test_idx,1]

    test_label = np.vstack((test_idx,test_target)).transpose()

    #Creates a list containing the train, validation, and test labels.
    labels = [train_label,valid_label,test_label]

    with open(datadir+'labels.pkl', 'wb') as f:
      pickle.dump(labels, f)

    args = MyClass()

    args.num_layers=int(5) #number of GT/FastGT layers
    args.lr=float(0.01) #learning rate
    args.non_local_weight=float(-2) #weight initialization for non local operations
    args.K=int(9) #number of non-local negibors
    args.dataset='Null' #Dataset
    args.model='FastGTN' #Model
    args.epoch=int(10) #Training Epochs
    args.channel_agg='mean'
    args.num_channels=int(2) #number of channels
    args.non_local=True #use non local operations
    args.node_dim=len(types)+1 #hidden dimensions
    args.weight_decay=float(0.001) #l2 reg
    args.runs=int(2) #number of runs
    args.remove_self_loops=False #remove_self_loops
    args.beta=float(0) #beta (Identity matrix)
    args.pre_train=False #pre-training FastGT layers
    args.num_FastGTN_layers=int(2) #number of FastGTN layers
    args.cluster=os.path.basename(fname)

    #subprocess.call(['python', 'main.py', '--dataset', 'Null', '--model', 'FastGTN', '--num_layers', '3', '--epoch', '200', '--lr', '0.04', '--channel_agg', 'mean', '--num_channels', '2', '--non_local_weight', '-2', '--K', '3', '--non_local', '--cluster', fname])
    trainTheModel(args)

