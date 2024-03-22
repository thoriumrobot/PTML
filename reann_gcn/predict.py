import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np
import os
import subprocess
import sys
from json_tricks import loads
import glob
from torch.utils.data import DataLoader

os.environ['DGLBACKEND'] = 'pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

primitive_types=["void", "byte", "short", "int", "long", "float", "double", "char", "boolean", "voidModifier", "byteModifier", "shortModifier", "intModifier", "longModifier", "floatModifier", "doubleModifier", "charModifier", "booleanModifier", "NonNullMarker", "final Modifier", "ArrayType", "ClassOrInterfaceType", "VariableDeclarationExpr"]
nodeList=["MethodDeclaration", "FieldDeclaration"]
types=['CompilationUnit', 'PackageDeclaration', 'Name', 'ImportDeclaration', 'ClassOrInterfaceDeclaration', 'Modifier', 'SimpleName', 'FieldDeclaration', 'intModifier', 'public Modifier', 'static Modifier', 'final Modifier', 'VariableDeclarator', 'PrimitiveType', 'UnaryExpr', 'IntegerLiteralExpr', 'ClassOrInterfaceType', 'StringLiteralExpr', 'AnnotationDeclaration', 'NormalAnnotationExpr', 'MemberValuePair', 'ArrayInitializerExpr', 'NameExpr', 'private Modifier', 'booleanModifier', 'ConstructorDeclaration', 'Parameter', 'MarkerAnnotationExpr', 'BackTargetTypeMarker', 'BlockStmt', 'ExpressionStmt', 'AssignExpr', 'NonNullMarker', 'MethodCallExpr', 'FieldAccessExpr', 'MethodDeclaration', 'OverrideMarker', 'VoidType', 'ReturnStmt', 'IfStmt', 'BinaryExpr', 'NullLiteralExpr', 'VariableDeclarationExpr', 'ObjectCreationExpr', 'ArrayType', 'ArrayCreationExpr', 'ArrayCreationLevel', 'CharLiteralExpr', 'SwitchStmt', 'SwitchEntry', 'SingleMemberAnnotationExpr', 'ThisExpr', 'SystemApiMarker', 'abstract Modifier', 'DeprecatedMarker', 'ConfigFlagMarker', 'BooleanLiteralExpr', 'ThrowStmt', 'byteModifier', 'shortModifier', 'ForEachStmt', 'UnsupportedAppUsageMarker', 'floatModifier', 'TryStmt', 'ConditionalExpr', 'ExplicitConstructorInvocationStmt', 'CastExpr', 'Vr2dDisplayFlagMarker', 'EnclosedExpr', 'longModifier', 'BreakStmt', 'BlockComment', 'InstanceOfExpr', 'ProfileTypeMarker', 'CallbackExecutorMarker', 'LambdaExpr', 'charModifier', 'ForStmt', 'ClassExpr', 'doubleModifier', 'ArrayAccessExpr', 'WhileStmt', 'SuperExpr', 'AttributionFlagsMarker', 'default Modifier', 'ChangeIdMarker', 'FlagsMarker', 'InputConfigFlagsMarker', 'SynchronizedStmt', 'LineComment', 'TypeParameter', 'WildcardType', 'UserIdIntMarker', 'UnknownType', 'SysUISingletonMarker', 'InjectMarker', 'ContinueStmt', 'synchronized Modifier', 'native Modifier', 'protected Modifier', 'TypeMarker', 'CallSuperMarker', 'NotifyEventTypeMarker', 'InitializerDeclaration', 'ResultMarker', 'volatile Modifier', 'VisibleForTestingMarker', 'LongLiteralExpr', 'TestApiMarker', 'IpcDataCacheModuleMarker', 'WorkerThreadMarker', 'JavadocComment', 'NetworkTypeMarker', 'CurrentTimeMillisLongMarker', 'ColorIntMarker', 'ColorLongMarker', 'DoubleLiteralExpr', 'GameModeMarker', 'UserHandleAwareMarker', 'MethodReferenceExpr', 'TypeExpr', 'RenderModeMarker', 'PermissionTypeMarker', 'MemberMarker', 'AnyThreadMarker', 'InputMethodNavButtonFlagsMarker', 'BreakStrategyMarker', 'HyphenationFrequencyMarker', 'JustificationModeMarker', 'PxMarker', 'FastNativeMarker', 'CriticalNativeMarker', 'OriginEnumMarker', 'PurposeEnumMarker', 'EncryptionPaddingEnumMarker', 'SignaturePaddingEnumMarker', 'DigestEnumMarker', 'BlockModeEnumMarker', 'AuthEnumMarker', 'SecurityLevelEnumMarker', 'MainMarker', 'LetterboxHorizontalReachabilityPositionMarker', 'LetterboxVerticalReachabilityPositionMarker', 'OriginMarker', 'InsetsTypeMarker', 'DispatchModeMarker', 'SecurityPatchStateMarker', 'LevelMarker', 'KeyAlgorithmEnumMarker', 'StateMarker', 'AutofillTypeMarker', 'RotationMarker', 'VibrationIntensityMarker', 'StringResMarker', 'AttrResMarker', 'BytesLongMarker', 'PartitionTypeMarker', 'AppearanceMarker', 'ActionTypeMarker', 'FlagTypeMarker', 'RequestFlagsMarker', 'AnimationTypeMarker', 'transient Modifier', 'UiThreadMarker', 'AssertStmt', 'ActivityTypeMarker', 'AvailabilityMarker', 'RequestTemplateMarker', 'ErrorCodeMarker', 'CAMERA_AUDIO_RESTRICTIONMarker', 'CapabilityStateMarker', 'MainThreadMarker', 'ImmutableMarker', 'SamplingStrategyMarker', 'EnumDeclaration', 'EnumConstantDeclaration', 'KeyguardBouncerScopeMarker', 'LockoutModeMarker', 'DrawableResMarker', 'IconTypeMarker', 'ChangeTypeMarker', 'SettingMarker', 'TransitionOldTypeMarker', 'RemoteViewMarker', 'StyleResMarker', 'RemotableViewMethodMarker', 'RecognitionFlagsMarker', 'ConfigMarker', 'ImplementationMarker', 'DirectMarker', 'StatusMarker', 'RuleMarker', 'BeforeMarker', 'AfterMarker', 'TestMarker', 'ViewportTypeMarker', 'EnrollReasonMarker', 'SensorTypeMarker', 'ElapsedRealtimeLongMarker', 'EmptyStmt', 'StaticMarker', 'WindowingModeMarker', 'PriorityMarker', 'ConnectorMarker', 'PermissionInfoFlagsMarker', 'PermissionWhitelistFlagsMarker', 'ProvidesMarker', 'CentralSurfacesScopeMarker', 'BindsMarker', 'IntoSetMarker', 'ConnectionToSinkTypeMarker', 'QueryFlagsMarker', 'EventTypesFlagMarker', 'CiphersuiteMarker', 'TransitionTypeMarker', 'TranslationFlagMarker', 'ApplyStrategyMarker', 'OrientationMarker', 'RequestTypeMarker', 'EventTypeMarker', 'ReadModeMarker', 'TransitionDirectionMarker', 'DoStmt', 'DocumentedMarker', 'DurationMillisLongMarker', 'SilentHeaderMarker', 'DismissalSurfaceMarker', 'DismissalSentimentMarker', 'ResponseResultMarker', 'DataFormatMarker', 'WriteModeMarker', 'StartResultMarker', 'StartArgFlagsMarker', 'StopForegroundFlagsMarker', 'EventMarker', 'AuthorizationStateMarker', 'ForegroundServiceTypeMarker', 'WakeReasonMarker', 'GoToSleepReasonMarker', 'ResultCodeMarker', 'PresubmitMarker', 'SmallTestMarker', 'BinderThreadMarker', 'TemplateTypeMarker', 'FormatMarker', 'LargeTestMarker', 'UiThreadTestMarker', 'ResponseCodeMarker', 'SessionModeMarker', 'SendRequestMarker', 'SendResultMarker', 'UiTemplateTypeMarker', 'CardStateInfoMarker', 'CheckResultMarker', 'ShortcutTypeMarker', 'AccessibilityFragmentTypeMarker', 'CinematicEffectStatusCodeMarker', 'ImageContentTypeMarker', 'StandbyBucketsMarker', 'ForcedReasonsMarker', 'ProcessStateMarker', 'AppActionMarker', 'AttestationProfileIdMarker', 'ViewModeMarker', 'ServiceStatusMarker', 'WarningTypeMarker', 'LayoutlibDelegateMarker', 'MissingMethodFlagsMarker', '/*package*/\nlongModifier', 'SearchIndexableMarker', 'MockMarker', 'CheckForNullMarker', 'NullableDeclMarker', 'NullableTypeMarker', 'NullAllowedMarker', 'NullUnknownMarker', 'NonnullMarker', 'NotNullMarker', 'NonNullDeclMarker', 'NonNullTypeMarker', 'NonNullByDefaultMarker', 'ParametersAreNonnullByDefaultMarker', 'ArrayResMarker', 'IgnoreMarker', 'StabilityMarker', 'PreciseCallStatesMarker', 'NrVopsStatusMarker', 'NrEmcStatusMarker', 'ImsStateMarker', 'ImsServiceCapabilityMarker', 'TransportTypeMarker', 'ExternalCallStateMarker', 'ImsRegistrationTechMarker', 'FeatureTypeMarker', 'PositionMarker', 'AppTypeMarker', 'IntRangeMarker', 'BiopMessageTypeMarker', 'LayoutResMarker', 'ThrottlingStatusMarker', 'KeyEventActionMarker', 'GwpAsanModeMarker', 'MemtagModeMarker', 'NativeHeapZeroInitializedMarker', 'StagedOperationTypeMarker', 'DistroStatusMarker', 'NotifyFlagsMarker', 'MotionEventActionMarker', 'ExtconDeviceTypeMarker', 'CallAudioRouteMarker', 'DeviceConfigKeyMarker', 'EventCategoryMarker', 'LetterboxBackgroundTypeMarker', 'LetterboxReachabilityPositionMarker', 'VariantMarker', 'ViewTypeMarker', 'FunctionalInterfaceMarker', 'RepeatModeMarker', 'BackgroundMarker', 'DimensionMarker', 'ColorResMarker', 'RawResMarker', 'ReactMethodMarker', 'GravityFlagMarker', 'LocalClassDeclarationStmt', 'AnimResMarker', 'GetterMarker', 'Slf4jMarker', 'SetterMarker', 'IdResMarker', 'screenTypeMarker', 'AutoAccessMarker', 'LightnessMarker', 'CalledByNativeMarker', 'AnnotationMemberDeclaration', 'NotThreadSafeMarker', 'ThreadSafeMarker', 'ErrorTypeMarker', 'JavascriptInterfaceMarker', 'SigninAccessPointMarker', 'ModalDialogTypeMarker', 'SearchEnginePromoTypeMarker', 'ReauthScopeMarker', 'OverrideStateMarker', 'SafeVarargsMarker', 'MediumTestMarker', 'EncodingMarker', 'TextSizeTypeMarker', 'ResizeModeMarker', 'StreamTypeMarker', 'StereoModeMarker', 'KeepMarker', 'BeforeClassMarker', 'FlashModeMarker', 'SubscribeMarker', 'MenuResMarker', 'AnimatorResMarker', 'AutoRestoreMarker', 'SingletonMarker', 'StatusCodeMarker', 'ActivityScopeMarker', 'LabeledStmt', 'KeycodeMarker', 'DraggableItemStateFlagsMarker', 'ScrollDirectionMarker', 'DimenResMarker', 'InternalApiMarker', 'JsonCreatorMarker', 'JsonIgnoreMarker', 'JsonPropertyMarker', 'DoNotStripMarker', 'UIManagerTypeMarker', 'ImageEventTypeMarker', 'ModeMarker', 'WMSingletonMarker', 'ShellMainThreadMarker', 'PropMarker', 'LinearColorMarker', 'EntityMarker', 'EntityInstanceMarker', 'CarProtocolMarker', 'KeepFieldsMarker', 'CarIconTypeMarker', 'ExperimentalCarApiMarker', 'CarColorTypeMarker', 'DoNotInlineMarker', 'AutoValueMarker', 'CarZoneRowMarker', 'CarZoneColumnMarker', 'StyleableResMarker', 'NonParcelFieldMarker', 'InputModeMarker', 'ReplaceStrategyMarker', 'ImageModeMarker', 'EvConnectorTypeMarker', 'FuelTypeMarker', 'NestedScrollTypeMarker', 'HorizontalAlignmentMarker', 'CommandVersionMarker', 'BuilderMarker', 'CanIgnoreReturnValueMarker', 'ProtoLayoutExperimentalMarker', 'SplitFinishBehaviorMarker', 'ExtLayoutDirectionMarker', 'RatioMarker', 'ComplicationTypeMarker', 'SplitPlaceholderFinishBehaviorMarker', 'AppCompatShadowedAttributesMarker', 'LongPropertyMarker', 'StringPropertyMarker', 'IdMarker', 'HvacFanDirectionMarker', 'AccessErrorMarker', 'VarType', 'TabSelectionTypeMarker', 'NativeMethodsMarker', 'InitializerMarker', 'SectionTypeMarker', 'GETMarker', 'UrlMarker', 'BodyMarker', 'ValueMarker', 'VideoProjectionFlagsMarker', 'SyntheticMarker', 'PureMarker', 'LogLevelMarker', 'PlaybackSuppressionReasonMarker', 'MessageTypeMarker', 'MonotonicNonNullMarker', 'TrackTypeMarker', 'CapabilitiesMarker', 'ColorSpaceMarker', 'ColorRangeMarker', 'ColorTransferMarker', 'VolumeFlagsMarker', 'VideoOutputModeMarker', 'RequirementFlagsMarker', 'AudioContentTypeMarker', 'AudioFlagsMarker', 'AudioUsageMarker', 'AudioAllowedCapturePolicyMarker', 'SpatializationBehaviorMarker', 'PcmEncodingMarker', 'TabCreationStateMarker', 'SecureModeMarker', 'TabLaunchTypeMarker', 'BufferReplacementModeMarker', 'BindsOptionalOfMarker', 'OptionalBooleanMarker', 'FontSizeUnitMarker', 'ExperimentalMarker', 'PropDefaultMarker', 'DirtinessStateMarker', 'AdaptiveToolbarButtonVariantMarker', 'StateChangeReasonMarker', 'ExpoMethodMarker', 'HiddenApiMarker', 'SlowMarker', 'ServiceMarker', 'IncubatingMarker', 'BetaMarker', 'TaskActionMarker', 'InputMarker', 'OptionalMarker', 'OutputDirectoryMarker', 'InputFilesMarker', 'NestedMarker', 'NonNlsMarker', 'InternalMarker', 'ModifierConstantMarker', 'TestOnlyMarker', 'XmlTransientMarker', 'InputFileMarker', 'ClassRuleMarker', 'AfterClassMarker', 'ParameterMarker', 'MemoizedMarker', 'AssistedMarker', 'GerritServerConfigMarker', 'SendEmailExecutorMarker', 'GerritPersonIdentMarker', 'AssistedInjectMarker', 'UiFieldMarker', 'MorphiaInternalMarker', 'SubstituteMarker', 'AliasMarker', 'ParameterizedTestMarker', 'BeforeEachMarker', 'AfterEachMarker', 'NewFieldMarker', 'TraceMarker', 'FieldNameMarker', 'DataMarker', 'ComponentMarker', 'RequiredArgsConstructorMarker', 'ToStringMarker', 'XmlElementMarker', 'ColumnMarker', 'ValidMarker', 'ManagedDataMarker', 'ManagedAttributeMarker', 'NoArgsConstructorMarker', 'AllArgsConstructorMarker', 'ControllerMarker', 'AutowiredMarker', 'ExtensionMarker', 'DataBoundConstructorMarker', 'RequirePOSTMarker', 'ExportedBeanMarker', 'DataBoundSetterMarker', 'QueryParameterMarker', 'PrivateMarker', 'TestExtensionMarker', 'CheckReturnValueMarker', 'AncestorInPathMarker', 'POSTMarker', 'UtilityClassMarker', 'WhitelistedMarker', 'PostConstructMarker', 'SneakyThrowsMarker', 'EqualsAndHashCodeMarker', 'TransientMarker', 'DefaultMarker', 'RequiredMarker', 'RestControllerMarker', 'NonNullApiMarker', 'NonNullFieldsMarker', 'ParamMarker', 'HeaderMarker', 'RequestParamMarker', 'ValidatedMarker', 'NonnegativeMarker', 'JsonpDeserializableMarker', 'EditableMarker', 'NotEmptyMarker', 'BeforeAllMarker', 'AfterAllMarker', 'EvolvingMarker', 'GwtCompatibleMarker', 'WeakMarker', 'GwtIncompatibleMarker', 'WeakOuterMarker', 'ViewComponentMarker', 'J2ObjCIncompatibleMarker', '/* static */\nlongModifier', 'ForOverrideMarker', 'DerivedMarker', 'CheckMarker', 'AddToRuleKeyMarker', 'BuckStyleValueMarker', 'OnChannelThreadMarker', 'OnClientThreadMarker', 'UnknownKeyForMarker', 'InitializedMarker', 'ProcessElementMarker', 'ElementMarker', 'WithBeanGetterMarker', 'JsonAutoDetectMarker', 'ObjectIdMarker', 'WithSpanMarker', 'ConfigurationMarker', 'NotBlankMarker', 'ContextMarker', 'TimedMarker', 'DELETEMarker', 'PositiveMarker', 'PositiveOrZeroMarker', 'AlphaMarker', 'AccessesPartialKeyMarker', 'AutoCodecMarker', 'InstantiatorMarker', 'VisibleForSerializationMarker', 'SerializationConstantMarker', 'StarlarkConstructorMarker', 'NamedMarker', 'PublicEvolvingMarker', 'RpcTimeoutMarker', 'BenchmarkMarker', 'NullFromTypeParamMarker', 'UnmodifiableMarker', 'ReferenceMarker', 'SerialMarker', 'ActivateMarker', 'DeactivateMarker', 'JaxrsResourceMarker', 'JSONRequiredMarker', 'WebSocketMarker', 'OnWebSocketCloseMarker', 'OnWebSocketConnectMarker', 'OnWebSocketMessageMarker', 'OnWebSocketErrorMarker', 'ModifiedMarker', 'ExposeMarker', 'PreDestroyMarker', 'EventHandlerMarker', 'NlsSafeMarker', 'NlsMarker', 'ExcludeMarker', 'ShadowMarker', 'TransactionalMarker', 'FinalDefaultMarker', 'ConcurrentMethodMarker', 'OverridingMethodsMustInvokeSuperMarker', 'DialogTitleMarker', 'Log4j2Marker', 'BeanMarker', 'ResourceMarker', 'TooltipMarker', 'DialogMessageMarker', 'ButtonMarker', 'StubbedMarker', 'NotificationTitleMarker', 'ProgressTitleMarker', 'ActionTextMarker', 'InspectionMessageMarker', 'NotificationContentMarker', 'IntentionFamilyNameMarker', 'IntentionNameMarker', 'SafeFieldForPreviewMarker', 'SupportMarker', 'JsNonNullMarker', 'NullMarkedMarker', 'KtPropertyMarker', 'AutoConfigurationMarker', 'JmixPropertyMarker', 'RequestBodyMarker', 'ReadOperationMarker', 'SelectorMarker', 'ParameterizedAdapterTestMarker', 'PathVariableMarker', 'GetExchangeMarker', 'TestValueMarker', 'EnableCachingMarker', 'ParameterizedHttpServerTestMarker', 'PrimaryMarker', 'ConditionalOnMissingBeanMarker', 'ProgrammaticMarker', 'SpringIntegrationTestMarker', 'CreatedDateMarker', 'LastModifiedDateMarker', 'EnableBatchProcessingMarker', 'RepositoryMarker', 'MemberSupportMarker', 'GraphQlExceptionHandlerMarker', 'VolatileMarker', 'CopyMarker', 'InitMarker', 'Other']
token_to_id = {token: i for i, token in enumerate(types)}

# Initialize the encoder with the complete list of node types
def multi_hot_encode(node, token_to_id):
    encoding = [0] * len(token_to_id)
    for token in node['type']:  # Assuming 'type' is the list with node type and modifiers
        if token in types:
            encoding[token_to_id[token]] = 1
        else:
            encoding[token_to_id['Other']] = 1
    return encoding

def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(labels, dim=0)

def ast_to_dgl(ast_graph, encoder):
    # Create a graph with the given number of nodes
    g = dgl.graph(([], []))
    g.add_nodes(len(ast_graph['nodes']))
    
    num_nodes = len(ast_graph['nodes'])
    
    # Add edges based on the adjacency list
    for src, dst_list in ast_graph['adjacencyList'].items():
        valid_dst = [dst for dst in dst_list if dst < num_nodes]
        for dst in valid_dst:
            g.add_edges(int(src), dst)
    
    # Extract node types and multi-hot encode them
    node_features = [multi_hot_encode(node, token_to_id) for node in ast_graph['nodes']]
    
    # Set node features
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
    
    # Extract labels (whether a node is nullable or not)
    labels = torch.tensor([node['nullable'] for node in ast_graph['nodes']], dtype=torch.long)
    
    return g, labels

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Set input feature
        g.ndata['h'] = feature
        # Message passing
        g.update_all(fn.copy_u('h', 'm'), fn.mean(msg='m', out='h'))
        # Apply non-linearity
        h = g.ndata['h']
        return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.gcn1(g, features))
        x = self.gcn2(g, x)
        return x

modeldir='/home/ubuntu/overkillisunderrated/nullproj/reann_gcn/'

batch_size=10
epochs=20

fname=os.path.join(modeldir, "temp_output.json")

json_data = []

# Load JSON data from file
with open(fname, "r") as file:
  json_data = json.load(file)

json_data=[json_data]

# Assume g is your DGL graph object and features is your node features tensor
model = GCN(in_feats=len(types), hidden_size=16, num_classes=2).to(device)  # Binary classification
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_loss = float('inf')

with open(modeldir+"best_model_"+sys.argv[1]+".json.pkl",'rb') as f:
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint)
    model = model.to(device)

graphs_and_labels = [ast_to_dgl(graph, token_to_id) for graph in json_data]

# Collate the graphs and labels
batched_graph, labels = collate(graphs_and_labels)

batched_graph = batched_graph.to(device)
labels = labels.to(device)

# Extract the features from the batched graph
features = batched_graph.ndata['feat'].to(device)

model = model.eval()

with torch.no_grad():
    logits = model(batched_graph, features)
    
predictions = torch.argmax(logits, dim=1)

# Get the indices where the value is 1
indices = torch.nonzero(predictions == 1).squeeze().tolist()
for index in indices:
    node_types=json_data[0]['nodes'][index]['type']
    if ("MethodDeclaration" in node_types) or ("FieldDeclaration" in node_types):
        print(index)
