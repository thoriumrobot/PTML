import networkx as nx
import numpy as np
from scipy.sparse.csgraph import laplacian
#from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import json
import os
from scipy.spatial import distance
from json_tricks import loads
import pickle

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

types_list=['CompilationUnit', 'PackageDeclaration', 'Name', 'ImportDeclaration', 'ClassOrInterfaceDeclaration', 'Modifier', 'SimpleName', 'FieldDeclaration', 'intModifier', 'public Modifier', 'static Modifier', 'final Modifier', 'VariableDeclarator', 'PrimitiveType', 'UnaryExpr', 'IntegerLiteralExpr', 'ClassOrInterfaceType', 'StringLiteralExpr', 'AnnotationDeclaration', 'NormalAnnotationExpr', 'MemberValuePair', 'ArrayInitializerExpr', 'NameExpr', 'private Modifier', 'booleanModifier', 'ConstructorDeclaration', 'Parameter', 'MarkerAnnotationExpr', 'BackTargetTypeMarker', 'BlockStmt', 'ExpressionStmt', 'AssignExpr', 'NonNullMarker', 'MethodCallExpr', 'FieldAccessExpr', 'MethodDeclaration', 'void', 'OverrideMarker', 'VoidType', 'ReturnStmt', 'IfStmt', 'BinaryExpr', 'NullLiteralExpr', 'VariableDeclarationExpr', 'ObjectCreationExpr', 'ArrayType', 'ArrayCreationExpr', 'ArrayCreationLevel', 'CharLiteralExpr', 'SwitchStmt', 'SwitchEntry', 'SingleMemberAnnotationExpr', 'ThisExpr', 'SystemApiMarker', 'abstract Modifier', 'DeprecatedMarker', 'ConfigFlagMarker', 'BooleanLiteralExpr', 'ThrowStmt', 'byteModifier', 'shortModifier', 'ForEachStmt', 'UnsupportedAppUsageMarker', 'floatModifier', 'TryStmt', 'ConditionalExpr', 'ExplicitConstructorInvocationStmt', 'CastExpr', 'Vr2dDisplayFlagMarker', 'EnclosedExpr', 'longModifier', 'BreakStmt', 'BlockComment', 'InstanceOfExpr', 'ProfileTypeMarker', 'CallbackExecutorMarker', 'LambdaExpr', 'charModifier', 'ForStmt', 'ClassExpr', 'doubleModifier', 'ArrayAccessExpr', 'WhileStmt', 'SuperExpr', 'AttributionFlagsMarker', 'default Modifier', 'ChangeIdMarker', 'FlagsMarker', 'InputConfigFlagsMarker', 'SynchronizedStmt', 'LineComment', 'TypeParameter', 'WildcardType', 'UserIdIntMarker', 'UnknownType', 'SysUISingletonMarker', 'InjectMarker', 'ContinueStmt', 'synchronized Modifier', 'native Modifier', 'protected Modifier', 'TypeMarker', 'CallSuperMarker', 'NotifyEventTypeMarker', 'InitializerDeclaration', 'ResultMarker', 'volatile Modifier', 'VisibleForTestingMarker', 'LongLiteralExpr', 'TestApiMarker', 'IpcDataCacheModuleMarker', 'WorkerThreadMarker', 'JavadocComment', 'NetworkTypeMarker', 'CurrentTimeMillisLongMarker', 'ColorIntMarker', 'ColorLongMarker', 'DoubleLiteralExpr', 'GameModeMarker', 'UserHandleAwareMarker', 'MethodReferenceExpr', 'TypeExpr', 'RenderModeMarker', 'PermissionTypeMarker', 'MemberMarker', 'AnyThreadMarker', 'InputMethodNavButtonFlagsMarker', 'BreakStrategyMarker', 'HyphenationFrequencyMarker', 'JustificationModeMarker', 'PxMarker', 'FastNativeMarker', 'CriticalNativeMarker', 'OriginEnumMarker', 'PurposeEnumMarker', 'EncryptionPaddingEnumMarker', 'SignaturePaddingEnumMarker', 'DigestEnumMarker', 'BlockModeEnumMarker', 'AuthEnumMarker', 'SecurityLevelEnumMarker', 'MainMarker', 'LetterboxHorizontalReachabilityPositionMarker', 'LetterboxVerticalReachabilityPositionMarker', 'OriginMarker', 'InsetsTypeMarker', 'DispatchModeMarker', 'SecurityPatchStateMarker', 'LevelMarker', 'KeyAlgorithmEnumMarker', 'StateMarker', 'AutofillTypeMarker', 'RotationMarker', 'VibrationIntensityMarker', 'StringResMarker', 'AttrResMarker', 'BytesLongMarker', 'PartitionTypeMarker', 'AppearanceMarker', 'ActionTypeMarker', 'FlagTypeMarker', 'RequestFlagsMarker', 'AnimationTypeMarker', 'transient Modifier', 'UiThreadMarker', 'AssertStmt', 'ActivityTypeMarker', 'AvailabilityMarker', 'RequestTemplateMarker', 'ErrorCodeMarker', 'CAMERA_AUDIO_RESTRICTIONMarker', 'CapabilityStateMarker', 'MainThreadMarker', 'ImmutableMarker', 'SamplingStrategyMarker', 'EnumDeclaration', 'EnumConstantDeclaration', 'KeyguardBouncerScopeMarker', 'LockoutModeMarker', 'DrawableResMarker', 'IconTypeMarker', 'ChangeTypeMarker', 'SettingMarker', 'TransitionOldTypeMarker', 'RemoteViewMarker', 'StyleResMarker', 'RemotableViewMethodMarker', 'RecognitionFlagsMarker', 'ConfigMarker', 'ImplementationMarker', 'DirectMarker', 'StatusMarker', 'RuleMarker', 'BeforeMarker', 'AfterMarker', 'TestMarker', 'ViewportTypeMarker', 'EnrollReasonMarker', 'SensorTypeMarker', 'ElapsedRealtimeLongMarker', 'EmptyStmt', 'StaticMarker', 'WindowingModeMarker', 'PriorityMarker', 'ConnectorMarker', 'PermissionInfoFlagsMarker', 'PermissionWhitelistFlagsMarker', 'ProvidesMarker', 'CentralSurfacesScopeMarker', 'BindsMarker', 'IntoSetMarker', 'ConnectionToSinkTypeMarker', 'QueryFlagsMarker', 'EventTypesFlagMarker', 'CiphersuiteMarker', 'TransitionTypeMarker', 'TranslationFlagMarker', 'ApplyStrategyMarker', 'OrientationMarker', 'RequestTypeMarker', 'EventTypeMarker', 'ReadModeMarker', 'TransitionDirectionMarker', 'DoStmt', 'DocumentedMarker', 'DurationMillisLongMarker', 'SilentHeaderMarker', 'DismissalSurfaceMarker', 'DismissalSentimentMarker', 'ResponseResultMarker', 'DataFormatMarker', 'WriteModeMarker', 'StartResultMarker', 'StartArgFlagsMarker', 'StopForegroundFlagsMarker', 'EventMarker', 'AuthorizationStateMarker', 'ForegroundServiceTypeMarker', 'WakeReasonMarker', 'GoToSleepReasonMarker', 'ResultCodeMarker', 'PresubmitMarker', 'SmallTestMarker', 'BinderThreadMarker', 'TemplateTypeMarker', 'FormatMarker', 'LargeTestMarker', 'UiThreadTestMarker', 'ResponseCodeMarker', 'SessionModeMarker', 'SendRequestMarker', 'SendResultMarker', 'UiTemplateTypeMarker', 'CardStateInfoMarker', 'CheckResultMarker', 'ShortcutTypeMarker', 'AccessibilityFragmentTypeMarker', 'CinematicEffectStatusCodeMarker', 'ImageContentTypeMarker', 'StandbyBucketsMarker', 'ForcedReasonsMarker', 'ProcessStateMarker', 'AppActionMarker', 'AttestationProfileIdMarker', 'ViewModeMarker', 'ServiceStatusMarker', 'WarningTypeMarker', 'LayoutlibDelegateMarker', 'MissingMethodFlagsMarker', '/*package*/\nlongModifier', 'SearchIndexableMarker', 'MockMarker', 'CheckForNullMarker', 'NullableDeclMarker', 'NullableTypeMarker', 'NullAllowedMarker', 'NullUnknownMarker', 'NonnullMarker', 'NotNullMarker', 'NonNullDeclMarker', 'NonNullTypeMarker', 'NonNullByDefaultMarker', 'ParametersAreNonnullByDefaultMarker', 'ArrayResMarker', 'IgnoreMarker', 'StabilityMarker', 'PreciseCallStatesMarker', 'NrVopsStatusMarker', 'NrEmcStatusMarker', 'ImsStateMarker', 'ImsServiceCapabilityMarker', 'TransportTypeMarker', 'ExternalCallStateMarker', 'ImsRegistrationTechMarker', 'FeatureTypeMarker', 'PositionMarker', 'AppTypeMarker', 'IntRangeMarker', 'BiopMessageTypeMarker', 'LayoutResMarker', 'ThrottlingStatusMarker', 'KeyEventActionMarker', 'GwpAsanModeMarker', 'MemtagModeMarker', 'NativeHeapZeroInitializedMarker', 'StagedOperationTypeMarker', 'DistroStatusMarker', 'NotifyFlagsMarker', 'MotionEventActionMarker', 'ExtconDeviceTypeMarker', 'CallAudioRouteMarker', 'DeviceConfigKeyMarker', 'EventCategoryMarker', 'LetterboxBackgroundTypeMarker', 'LetterboxReachabilityPositionMarker', 'VariantMarker', 'ViewTypeMarker', 'FunctionalInterfaceMarker', 'RepeatModeMarker', 'BackgroundMarker', 'DimensionMarker', 'ColorResMarker', 'RawResMarker', 'ReactMethodMarker', 'GravityFlagMarker', 'LocalClassDeclarationStmt', 'AnimResMarker', 'GetterMarker', 'Slf4jMarker', 'SetterMarker', 'IdResMarker', 'screenTypeMarker', 'AutoAccessMarker', 'LightnessMarker', 'CalledByNativeMarker', 'AnnotationMemberDeclaration', 'NotThreadSafeMarker', 'ThreadSafeMarker', 'ErrorTypeMarker', 'JavascriptInterfaceMarker', 'SigninAccessPointMarker', 'ModalDialogTypeMarker', 'SearchEnginePromoTypeMarker', 'ReauthScopeMarker', 'OverrideStateMarker', 'SafeVarargsMarker', 'MediumTestMarker', 'EncodingMarker', 'TextSizeTypeMarker', 'ResizeModeMarker', 'StreamTypeMarker', 'StereoModeMarker', 'KeepMarker', 'BeforeClassMarker', 'FlashModeMarker', 'SubscribeMarker', 'MenuResMarker', 'AnimatorResMarker', 'AutoRestoreMarker', 'SingletonMarker', 'StatusCodeMarker', 'ActivityScopeMarker', 'LabeledStmt', 'KeycodeMarker', 'DraggableItemStateFlagsMarker', 'ScrollDirectionMarker', 'DimenResMarker', 'InternalApiMarker', 'JsonCreatorMarker', 'JsonIgnoreMarker', 'JsonPropertyMarker', 'DoNotStripMarker', 'UIManagerTypeMarker', 'ImageEventTypeMarker', 'ModeMarker', 'WMSingletonMarker', 'ShellMainThreadMarker', 'PropMarker', 'LinearColorMarker', 'EntityMarker', 'EntityInstanceMarker', 'CarProtocolMarker', 'KeepFieldsMarker', 'CarIconTypeMarker', 'ExperimentalCarApiMarker', 'CarColorTypeMarker', 'DoNotInlineMarker', 'AutoValueMarker', 'CarZoneRowMarker', 'CarZoneColumnMarker', 'StyleableResMarker', 'NonParcelFieldMarker', 'InputModeMarker', 'ReplaceStrategyMarker', 'ImageModeMarker', 'EvConnectorTypeMarker', 'FuelTypeMarker', 'NestedScrollTypeMarker', 'HorizontalAlignmentMarker', 'CommandVersionMarker', 'BuilderMarker', 'CanIgnoreReturnValueMarker', 'ProtoLayoutExperimentalMarker', 'SplitFinishBehaviorMarker', 'ExtLayoutDirectionMarker', 'RatioMarker', 'ComplicationTypeMarker', 'SplitPlaceholderFinishBehaviorMarker', 'AppCompatShadowedAttributesMarker', 'LongPropertyMarker', 'StringPropertyMarker', 'IdMarker', 'HvacFanDirectionMarker', 'AccessErrorMarker', 'VarType', 'TabSelectionTypeMarker', 'NativeMethodsMarker', 'InitializerMarker', 'SectionTypeMarker', 'GETMarker', 'UrlMarker', 'BodyMarker', 'ValueMarker', 'VideoProjectionFlagsMarker', 'SyntheticMarker', 'PureMarker', 'LogLevelMarker', 'PlaybackSuppressionReasonMarker', 'MessageTypeMarker', 'MonotonicNonNullMarker', 'TrackTypeMarker', 'CapabilitiesMarker', 'ColorSpaceMarker', 'ColorRangeMarker', 'ColorTransferMarker', 'VolumeFlagsMarker', 'VideoOutputModeMarker', 'RequirementFlagsMarker', 'AudioContentTypeMarker', 'AudioFlagsMarker', 'AudioUsageMarker', 'AudioAllowedCapturePolicyMarker', 'SpatializationBehaviorMarker', 'PcmEncodingMarker', 'TabCreationStateMarker', 'SecureModeMarker', 'TabLaunchTypeMarker', 'BufferReplacementModeMarker', 'BindsOptionalOfMarker', 'OptionalBooleanMarker', 'FontSizeUnitMarker', 'ExperimentalMarker', 'PropDefaultMarker', 'DirtinessStateMarker', 'AdaptiveToolbarButtonVariantMarker', 'StateChangeReasonMarker', 'ExpoMethodMarker', 'HiddenApiMarker', 'SlowMarker', 'ServiceMarker', 'IncubatingMarker', 'BetaMarker', 'TaskActionMarker', 'InputMarker', 'OptionalMarker', 'OutputDirectoryMarker', 'InputFilesMarker', 'NestedMarker', 'NonNlsMarker', 'InternalMarker', 'ModifierConstantMarker', 'TestOnlyMarker', 'XmlTransientMarker', 'InputFileMarker', 'ClassRuleMarker', 'AfterClassMarker', 'ParameterMarker', 'MemoizedMarker', 'AssistedMarker', 'GerritServerConfigMarker', 'SendEmailExecutorMarker', 'GerritPersonIdentMarker', 'AssistedInjectMarker', 'UiFieldMarker', 'MorphiaInternalMarker', 'SubstituteMarker', 'AliasMarker', 'ParameterizedTestMarker', 'BeforeEachMarker', 'AfterEachMarker', 'NewFieldMarker', 'TraceMarker', 'FieldNameMarker', 'DataMarker', 'ComponentMarker', 'RequiredArgsConstructorMarker', 'ToStringMarker', 'XmlElementMarker', 'ColumnMarker', 'ValidMarker', 'ManagedDataMarker', 'ManagedAttributeMarker', 'NoArgsConstructorMarker', 'AllArgsConstructorMarker', 'ControllerMarker', 'AutowiredMarker', 'ExtensionMarker', 'DataBoundConstructorMarker', 'RequirePOSTMarker', 'ExportedBeanMarker', 'DataBoundSetterMarker', 'QueryParameterMarker', 'PrivateMarker', 'TestExtensionMarker', 'CheckReturnValueMarker', 'AncestorInPathMarker', 'POSTMarker', 'UtilityClassMarker', 'WhitelistedMarker', 'PostConstructMarker', 'SneakyThrowsMarker', 'EqualsAndHashCodeMarker', 'TransientMarker', 'DefaultMarker', 'RequiredMarker', 'RestControllerMarker', 'NonNullApiMarker', 'NonNullFieldsMarker', 'ParamMarker', 'HeaderMarker', 'RequestParamMarker', 'ValidatedMarker', 'NonnegativeMarker', 'JsonpDeserializableMarker', 'EditableMarker', 'NotEmptyMarker', 'BeforeAllMarker', 'AfterAllMarker', 'EvolvingMarker', 'GwtCompatibleMarker', 'WeakMarker', 'GwtIncompatibleMarker', 'WeakOuterMarker', 'ViewComponentMarker', 'J2ObjCIncompatibleMarker', '/* static */\nlongModifier', 'ForOverrideMarker', 'DerivedMarker', 'CheckMarker', 'AddToRuleKeyMarker', 'BuckStyleValueMarker', 'OnChannelThreadMarker', 'OnClientThreadMarker', 'UnknownKeyForMarker', 'InitializedMarker', 'ProcessElementMarker', 'ElementMarker', 'WithBeanGetterMarker', 'JsonAutoDetectMarker', 'ObjectIdMarker', 'WithSpanMarker', 'ConfigurationMarker', 'NotBlankMarker', 'ContextMarker', 'TimedMarker', 'DELETEMarker', 'PositiveMarker', 'PositiveOrZeroMarker', 'AlphaMarker', 'AccessesPartialKeyMarker', 'AutoCodecMarker', 'InstantiatorMarker', 'VisibleForSerializationMarker', 'SerializationConstantMarker', 'StarlarkConstructorMarker', 'NamedMarker', 'PublicEvolvingMarker', 'RpcTimeoutMarker', 'BenchmarkMarker', 'NullFromTypeParamMarker', 'UnmodifiableMarker', 'ReferenceMarker', 'SerialMarker', 'ActivateMarker', 'DeactivateMarker', 'JaxrsResourceMarker', 'JSONRequiredMarker', 'WebSocketMarker', 'OnWebSocketCloseMarker', 'OnWebSocketConnectMarker', 'OnWebSocketMessageMarker', 'OnWebSocketErrorMarker', 'ModifiedMarker', 'ExposeMarker', 'PreDestroyMarker', 'EventHandlerMarker', 'NlsSafeMarker', 'NlsMarker', 'ExcludeMarker', 'ShadowMarker', 'TransactionalMarker', 'FinalDefaultMarker', 'ConcurrentMethodMarker', 'OverridingMethodsMustInvokeSuperMarker', 'DialogTitleMarker', 'Log4j2Marker', 'BeanMarker', 'ResourceMarker', 'TooltipMarker', 'DialogMessageMarker', 'ButtonMarker', 'StubbedMarker', 'NotificationTitleMarker', 'ProgressTitleMarker', 'ActionTextMarker', 'InspectionMessageMarker', 'NotificationContentMarker', 'IntentionFamilyNameMarker', 'IntentionNameMarker', 'SafeFieldForPreviewMarker', 'SupportMarker', 'JsNonNullMarker', 'NullMarkedMarker', 'KtPropertyMarker', 'AutoConfigurationMarker', 'JmixPropertyMarker', 'RequestBodyMarker', 'ReadOperationMarker', 'SelectorMarker', 'ParameterizedAdapterTestMarker', 'PathVariableMarker', 'GetExchangeMarker', 'TestValueMarker', 'EnableCachingMarker', 'ParameterizedHttpServerTestMarker', 'PrimaryMarker', 'ConditionalOnMissingBeanMarker', 'ProgrammaticMarker', 'SpringIntegrationTestMarker', 'CreatedDateMarker', 'LastModifiedDateMarker', 'EnableBatchProcessingMarker', 'RepositoryMarker', 'MemberSupportMarker', 'GraphQlExceptionHandlerMarker', 'VolatileMarker', 'CopyMarker', 'InitMarker']

max_size=0

def pad_adjacency_matrices(adjacency_matrices):
    global max_size

    # Pad each matrix to the maximum size
    padded_matrices = []
    for mat in adjacency_matrices:
        pad_size = max_size - mat.shape[0]
        padded_mat = np.pad(mat, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=-1)
        padded_matrices.append(padded_mat)

    return padded_matrices

directory = '/home/k/ks225/annotations/2hv/'
fname=os.path.join(directory, "output_prune.json")

json_data = []
with open(fname, "r") as file:
    while True:
        obj_str = getobj(file)
        if not obj_str:
           break
        json_obj = loads(obj_str)
        # Find the size of the largest matrix
        max_size=max(max_size,len(json_obj['nodes']))
        if len(json_obj['nodes'])>805 and len(json_obj['nodes'])<895:
            json_data.append(json_obj)

print("max_size = ", max_size)

print(len(json_data))

nnode=0
nmax=0
for graph_json in json_data:
   nodes=graph_json['nodes']
   for node in nodes:
     nnode+=1
     nmax=max(nmax, node['id'])

print('nnode = ',nnode)
print('nmax = ',nmax)

nterm=0

for graph_json in json_data:
   nterm+=len(graph_json['nameList'])

print('nterm = ',nterm)

nnum=nnode+nterm
print("nnum = ",nnum)

A_n=[]

for graph_json in json_data:
  nodes=graph_json['nodes']
  alist=graph_json['adjacencyList']
  amat=np.zeros((len(nodes), len(nodes)))
  for node in nodes:
    if str(node['id']) in alist:
      for neighbor in alist[str(node['id'])]:
        amat[node['id']-1,neighbor-1]=1
  A_n.append(amat)

# Given adjacency_matrices as an array of adjacency matrices
adjacency_matrices=pad_adjacency_matrices(A_n)

# Convert adjacency matrices to Laplacian matrices
laplacian_matrices = [laplacian(adj) for adj in adjacency_matrices]

# Flatten and stack Laplacian matrices for clustering
# (SpectralClustering expects 2D input)
laplacian_data = np.vstack([lap.flatten() for lap in laplacian_matrices])

# Perform KMeans clustering
# Here we assume 5 clusters, but you should tune this hyperparameter
clustering = KMeans(n_clusters=5, random_state=0).fit(laplacian_data)

with open('kmmodel.pkl', 'wb') as f:
  pickle.dump(clustering, f)

# Get the cluster labels for each AST
cluster_labels = clustering.labels_

print(cluster_labels)

# Get the centroids of the clusters
centroids = clustering.cluster_centers_

f=[]

for i in range(5):
   f.append(open('data'+str(i)+'.json', 'w'))

with open(fname, "r") as file:
    while True:
        obj_str = getobj(file)
        if not obj_str:
           break
        json_obj = json.loads(obj_str)

        nodes=json_obj['nodes']
        for node in nodes:
         if not any(t in types_list for t in node["type"]):
            node["type"].append("Other")

        alist=json_obj['adjacencyList']
        amat=np.zeros((len(nodes), len(nodes)))
        for node in nodes:
            if str(node['id']) in alist:
                for neighbor in alist[str(node['id'])]:
                    amat[node['id']-1,neighbor-1]=1

        # Transform new_matrix to a Laplacian matrix
        new_laplacian = laplacian(pad_adjacency_matrices([amat])[0])

        # Flatten the Laplacian matrix
        new_laplacian_flattened = new_laplacian.flatten()

        # Reshape the flattened matrix to fit the input shape of the KMeans model
        new_laplacian_reshaped = new_laplacian_flattened.reshape(1, -1)

        # Use the KMeans model to predict the cluster of the new matrix
        cluster_label = clustering.predict(new_laplacian_reshaped)

        f[cluster_label[0]].write(obj_str+'\n')

for i in range(5):
   f[i].close()
