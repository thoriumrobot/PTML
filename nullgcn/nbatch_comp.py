import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
import glob
from json_tricks import loads
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

primitive_types=["void", "byte", "short", "int", "long", "float", "double", "char", "boolean", "voidModifier", "byteModifier", "shortModifier", "intModifier", "longModifier", "floatModifier", "doubleModifier", "charModifier", "booleanModifier", "NonNullMarker", "final Modifier", "ArrayType", "ClassOrInterfaceType", "VariableDeclarationExpr"]
nodeList=["MethodDeclaration", "Parameter", "FieldDeclaration"]
types=['CompilationUnit', 'PackageDeclaration', 'Name', 'ImportDeclaration', 'ClassOrInterfaceDeclaration', 'Modifier', 'SimpleName', 'FieldDeclaration', 'intModifier', 'public Modifier', 'static Modifier', 'final Modifier', 'VariableDeclarator', 'PrimitiveType', 'UnaryExpr', 'IntegerLiteralExpr', 'ClassOrInterfaceType', 'StringLiteralExpr', 'AnnotationDeclaration', 'NormalAnnotationExpr', 'MemberValuePair', 'ArrayInitializerExpr', 'NameExpr', 'private Modifier', 'booleanModifier', 'ConstructorDeclaration', 'Parameter', 'MarkerAnnotationExpr', 'BackTargetTypeMarker', 'BlockStmt', 'ExpressionStmt', 'AssignExpr', 'NonNullMarker', 'MethodCallExpr', 'FieldAccessExpr', 'MethodDeclaration', 'OverrideMarker', 'VoidType', 'ReturnStmt', 'IfStmt', 'BinaryExpr', 'NullLiteralExpr', 'VariableDeclarationExpr', 'ObjectCreationExpr', 'ArrayType', 'ArrayCreationExpr', 'ArrayCreationLevel', 'CharLiteralExpr', 'SwitchStmt', 'SwitchEntry', 'SingleMemberAnnotationExpr', 'ThisExpr', 'SystemApiMarker', 'abstract Modifier', 'DeprecatedMarker', 'ConfigFlagMarker', 'BooleanLiteralExpr', 'ThrowStmt', 'byteModifier', 'shortModifier', 'ForEachStmt', 'UnsupportedAppUsageMarker', 'floatModifier', 'TryStmt', 'ConditionalExpr', 'ExplicitConstructorInvocationStmt', 'CastExpr', 'Vr2dDisplayFlagMarker', 'EnclosedExpr', 'longModifier', 'BreakStmt', 'BlockComment', 'InstanceOfExpr', 'ProfileTypeMarker', 'CallbackExecutorMarker', 'LambdaExpr', 'charModifier', 'ForStmt', 'ClassExpr', 'doubleModifier', 'ArrayAccessExpr', 'WhileStmt', 'SuperExpr', 'AttributionFlagsMarker', 'default Modifier', 'ChangeIdMarker', 'FlagsMarker', 'InputConfigFlagsMarker', 'SynchronizedStmt', 'LineComment', 'TypeParameter', 'WildcardType', 'UserIdIntMarker', 'UnknownType', 'SysUISingletonMarker', 'InjectMarker', 'ContinueStmt', 'synchronized Modifier', 'native Modifier', 'protected Modifier', 'TypeMarker', 'CallSuperMarker', 'NotifyEventTypeMarker', 'InitializerDeclaration', 'ResultMarker', 'volatile Modifier', 'VisibleForTestingMarker', 'LongLiteralExpr', 'TestApiMarker', 'IpcDataCacheModuleMarker', 'WorkerThreadMarker', 'JavadocComment', 'NetworkTypeMarker', 'CurrentTimeMillisLongMarker', 'ColorIntMarker', 'ColorLongMarker', 'DoubleLiteralExpr', 'GameModeMarker', 'UserHandleAwareMarker', 'MethodReferenceExpr', 'TypeExpr', 'RenderModeMarker', 'PermissionTypeMarker', 'MemberMarker', 'AnyThreadMarker', 'InputMethodNavButtonFlagsMarker', 'BreakStrategyMarker', 'HyphenationFrequencyMarker', 'JustificationModeMarker', 'PxMarker', 'FastNativeMarker', 'CriticalNativeMarker', 'OriginEnumMarker', 'PurposeEnumMarker', 'EncryptionPaddingEnumMarker', 'SignaturePaddingEnumMarker', 'DigestEnumMarker', 'BlockModeEnumMarker', 'AuthEnumMarker', 'SecurityLevelEnumMarker', 'MainMarker', 'LetterboxHorizontalReachabilityPositionMarker', 'LetterboxVerticalReachabilityPositionMarker', 'OriginMarker', 'InsetsTypeMarker', 'DispatchModeMarker', 'SecurityPatchStateMarker', 'LevelMarker', 'KeyAlgorithmEnumMarker', 'StateMarker', 'AutofillTypeMarker', 'RotationMarker', 'VibrationIntensityMarker', 'StringResMarker', 'AttrResMarker', 'BytesLongMarker', 'PartitionTypeMarker', 'AppearanceMarker', 'ActionTypeMarker', 'FlagTypeMarker', 'RequestFlagsMarker', 'AnimationTypeMarker', 'transient Modifier', 'UiThreadMarker', 'AssertStmt', 'ActivityTypeMarker', 'AvailabilityMarker', 'RequestTemplateMarker', 'ErrorCodeMarker', 'CAMERA_AUDIO_RESTRICTIONMarker', 'CapabilityStateMarker', 'MainThreadMarker', 'ImmutableMarker', 'SamplingStrategyMarker', 'EnumDeclaration', 'EnumConstantDeclaration', 'KeyguardBouncerScopeMarker', 'LockoutModeMarker', 'DrawableResMarker', 'IconTypeMarker', 'ChangeTypeMarker', 'SettingMarker', 'TransitionOldTypeMarker', 'RemoteViewMarker', 'StyleResMarker', 'RemotableViewMethodMarker', 'RecognitionFlagsMarker', 'ConfigMarker', 'ImplementationMarker', 'DirectMarker', 'StatusMarker', 'RuleMarker', 'BeforeMarker', 'AfterMarker', 'TestMarker', 'ViewportTypeMarker', 'EnrollReasonMarker', 'SensorTypeMarker', 'ElapsedRealtimeLongMarker', 'EmptyStmt', 'StaticMarker', 'WindowingModeMarker', 'PriorityMarker', 'ConnectorMarker', 'PermissionInfoFlagsMarker', 'PermissionWhitelistFlagsMarker', 'ProvidesMarker', 'CentralSurfacesScopeMarker', 'BindsMarker', 'IntoSetMarker', 'ConnectionToSinkTypeMarker', 'QueryFlagsMarker', 'EventTypesFlagMarker', 'CiphersuiteMarker', 'TransitionTypeMarker', 'TranslationFlagMarker', 'ApplyStrategyMarker', 'OrientationMarker', 'RequestTypeMarker', 'EventTypeMarker', 'ReadModeMarker', 'TransitionDirectionMarker', 'DoStmt', 'DocumentedMarker', 'DurationMillisLongMarker', 'SilentHeaderMarker', 'DismissalSurfaceMarker', 'DismissalSentimentMarker', 'ResponseResultMarker', 'DataFormatMarker', 'WriteModeMarker', 'StartResultMarker', 'StartArgFlagsMarker', 'StopForegroundFlagsMarker', 'EventMarker', 'AuthorizationStateMarker', 'ForegroundServiceTypeMarker', 'WakeReasonMarker', 'GoToSleepReasonMarker', 'ResultCodeMarker', 'PresubmitMarker', 'SmallTestMarker', 'BinderThreadMarker', 'TemplateTypeMarker', 'FormatMarker', 'LargeTestMarker', 'UiThreadTestMarker', 'ResponseCodeMarker', 'SessionModeMarker', 'SendRequestMarker', 'SendResultMarker', 'UiTemplateTypeMarker', 'CardStateInfoMarker', 'CheckResultMarker', 'ShortcutTypeMarker', 'AccessibilityFragmentTypeMarker', 'CinematicEffectStatusCodeMarker', 'ImageContentTypeMarker', 'StandbyBucketsMarker', 'ForcedReasonsMarker', 'ProcessStateMarker', 'AppActionMarker', 'AttestationProfileIdMarker', 'ViewModeMarker', 'ServiceStatusMarker', 'WarningTypeMarker', 'LayoutlibDelegateMarker', 'MissingMethodFlagsMarker', '/*package*/\nlongModifier', 'SearchIndexableMarker', 'MockMarker', 'CheckForNullMarker', 'NullableDeclMarker', 'NullableTypeMarker', 'NullAllowedMarker', 'NullUnknownMarker', 'NonnullMarker', 'NotNullMarker', 'NonNullDeclMarker', 'NonNullTypeMarker', 'NonNullByDefaultMarker', 'ParametersAreNonnullByDefaultMarker', 'ArrayResMarker', 'IgnoreMarker', 'StabilityMarker', 'PreciseCallStatesMarker', 'NrVopsStatusMarker', 'NrEmcStatusMarker', 'ImsStateMarker', 'ImsServiceCapabilityMarker', 'TransportTypeMarker', 'ExternalCallStateMarker', 'ImsRegistrationTechMarker', 'FeatureTypeMarker', 'PositionMarker', 'AppTypeMarker', 'IntRangeMarker', 'BiopMessageTypeMarker', 'LayoutResMarker', 'ThrottlingStatusMarker', 'KeyEventActionMarker', 'GwpAsanModeMarker', 'MemtagModeMarker', 'NativeHeapZeroInitializedMarker', 'StagedOperationTypeMarker', 'DistroStatusMarker', 'NotifyFlagsMarker', 'MotionEventActionMarker', 'ExtconDeviceTypeMarker', 'CallAudioRouteMarker', 'DeviceConfigKeyMarker', 'EventCategoryMarker', 'LetterboxBackgroundTypeMarker', 'LetterboxReachabilityPositionMarker', 'VariantMarker', 'ViewTypeMarker', 'FunctionalInterfaceMarker', 'RepeatModeMarker', 'BackgroundMarker', 'DimensionMarker', 'ColorResMarker', 'RawResMarker', 'ReactMethodMarker', 'GravityFlagMarker', 'LocalClassDeclarationStmt', 'AnimResMarker', 'GetterMarker', 'Slf4jMarker', 'SetterMarker', 'IdResMarker', 'screenTypeMarker', 'AutoAccessMarker', 'LightnessMarker', 'CalledByNativeMarker', 'AnnotationMemberDeclaration', 'NotThreadSafeMarker', 'ThreadSafeMarker', 'ErrorTypeMarker', 'JavascriptInterfaceMarker', 'SigninAccessPointMarker', 'ModalDialogTypeMarker', 'SearchEnginePromoTypeMarker', 'ReauthScopeMarker', 'OverrideStateMarker', 'SafeVarargsMarker', 'MediumTestMarker', 'EncodingMarker', 'TextSizeTypeMarker', 'ResizeModeMarker', 'StreamTypeMarker', 'StereoModeMarker', 'KeepMarker', 'BeforeClassMarker', 'FlashModeMarker', 'SubscribeMarker', 'MenuResMarker', 'AnimatorResMarker', 'AutoRestoreMarker', 'SingletonMarker', 'StatusCodeMarker', 'ActivityScopeMarker', 'LabeledStmt', 'KeycodeMarker', 'DraggableItemStateFlagsMarker', 'ScrollDirectionMarker', 'DimenResMarker', 'InternalApiMarker', 'JsonCreatorMarker', 'JsonIgnoreMarker', 'JsonPropertyMarker', 'DoNotStripMarker', 'UIManagerTypeMarker', 'ImageEventTypeMarker', 'ModeMarker', 'WMSingletonMarker', 'ShellMainThreadMarker', 'PropMarker', 'LinearColorMarker', 'EntityMarker', 'EntityInstanceMarker', 'CarProtocolMarker', 'KeepFieldsMarker', 'CarIconTypeMarker', 'ExperimentalCarApiMarker', 'CarColorTypeMarker', 'DoNotInlineMarker', 'AutoValueMarker', 'CarZoneRowMarker', 'CarZoneColumnMarker', 'StyleableResMarker', 'NonParcelFieldMarker', 'InputModeMarker', 'ReplaceStrategyMarker', 'ImageModeMarker', 'EvConnectorTypeMarker', 'FuelTypeMarker', 'NestedScrollTypeMarker', 'HorizontalAlignmentMarker', 'CommandVersionMarker', 'BuilderMarker', 'CanIgnoreReturnValueMarker', 'ProtoLayoutExperimentalMarker', 'SplitFinishBehaviorMarker', 'ExtLayoutDirectionMarker', 'RatioMarker', 'ComplicationTypeMarker', 'SplitPlaceholderFinishBehaviorMarker', 'AppCompatShadowedAttributesMarker', 'LongPropertyMarker', 'StringPropertyMarker', 'IdMarker', 'HvacFanDirectionMarker', 'AccessErrorMarker', 'VarType', 'TabSelectionTypeMarker', 'NativeMethodsMarker', 'InitializerMarker', 'SectionTypeMarker', 'GETMarker', 'UrlMarker', 'BodyMarker', 'ValueMarker', 'VideoProjectionFlagsMarker', 'SyntheticMarker', 'PureMarker', 'LogLevelMarker', 'PlaybackSuppressionReasonMarker', 'MessageTypeMarker', 'MonotonicNonNullMarker', 'TrackTypeMarker', 'CapabilitiesMarker', 'ColorSpaceMarker', 'ColorRangeMarker', 'ColorTransferMarker', 'VolumeFlagsMarker', 'VideoOutputModeMarker', 'RequirementFlagsMarker', 'AudioContentTypeMarker', 'AudioFlagsMarker', 'AudioUsageMarker', 'AudioAllowedCapturePolicyMarker', 'SpatializationBehaviorMarker', 'PcmEncodingMarker', 'TabCreationStateMarker', 'SecureModeMarker', 'TabLaunchTypeMarker', 'BufferReplacementModeMarker', 'BindsOptionalOfMarker', 'OptionalBooleanMarker', 'FontSizeUnitMarker', 'ExperimentalMarker', 'PropDefaultMarker', 'DirtinessStateMarker', 'AdaptiveToolbarButtonVariantMarker', 'StateChangeReasonMarker', 'ExpoMethodMarker', 'HiddenApiMarker', 'SlowMarker', 'ServiceMarker', 'IncubatingMarker', 'BetaMarker', 'TaskActionMarker', 'InputMarker', 'OptionalMarker', 'OutputDirectoryMarker', 'InputFilesMarker', 'NestedMarker', 'NonNlsMarker', 'InternalMarker', 'ModifierConstantMarker', 'TestOnlyMarker', 'XmlTransientMarker', 'InputFileMarker', 'ClassRuleMarker', 'AfterClassMarker', 'ParameterMarker', 'MemoizedMarker', 'AssistedMarker', 'GerritServerConfigMarker', 'SendEmailExecutorMarker', 'GerritPersonIdentMarker', 'AssistedInjectMarker', 'UiFieldMarker', 'MorphiaInternalMarker', 'SubstituteMarker', 'AliasMarker', 'ParameterizedTestMarker', 'BeforeEachMarker', 'AfterEachMarker', 'NewFieldMarker', 'TraceMarker', 'FieldNameMarker', 'DataMarker', 'ComponentMarker', 'RequiredArgsConstructorMarker', 'ToStringMarker', 'XmlElementMarker', 'ColumnMarker', 'ValidMarker', 'ManagedDataMarker', 'ManagedAttributeMarker', 'NoArgsConstructorMarker', 'AllArgsConstructorMarker', 'ControllerMarker', 'AutowiredMarker', 'ExtensionMarker', 'DataBoundConstructorMarker', 'RequirePOSTMarker', 'ExportedBeanMarker', 'DataBoundSetterMarker', 'QueryParameterMarker', 'PrivateMarker', 'TestExtensionMarker', 'CheckReturnValueMarker', 'AncestorInPathMarker', 'POSTMarker', 'UtilityClassMarker', 'WhitelistedMarker', 'PostConstructMarker', 'SneakyThrowsMarker', 'EqualsAndHashCodeMarker', 'TransientMarker', 'DefaultMarker', 'RequiredMarker', 'RestControllerMarker', 'NonNullApiMarker', 'NonNullFieldsMarker', 'ParamMarker', 'HeaderMarker', 'RequestParamMarker', 'ValidatedMarker', 'NonnegativeMarker', 'JsonpDeserializableMarker', 'EditableMarker', 'NotEmptyMarker', 'BeforeAllMarker', 'AfterAllMarker', 'EvolvingMarker', 'GwtCompatibleMarker', 'WeakMarker', 'GwtIncompatibleMarker', 'WeakOuterMarker', 'ViewComponentMarker', 'J2ObjCIncompatibleMarker', '/* static */\nlongModifier', 'ForOverrideMarker', 'DerivedMarker', 'CheckMarker', 'AddToRuleKeyMarker', 'BuckStyleValueMarker', 'OnChannelThreadMarker', 'OnClientThreadMarker', 'UnknownKeyForMarker', 'InitializedMarker', 'ProcessElementMarker', 'ElementMarker', 'WithBeanGetterMarker', 'JsonAutoDetectMarker', 'ObjectIdMarker', 'WithSpanMarker', 'ConfigurationMarker', 'NotBlankMarker', 'ContextMarker', 'TimedMarker', 'DELETEMarker', 'PositiveMarker', 'PositiveOrZeroMarker', 'AlphaMarker', 'AccessesPartialKeyMarker', 'AutoCodecMarker', 'InstantiatorMarker', 'VisibleForSerializationMarker', 'SerializationConstantMarker', 'StarlarkConstructorMarker', 'NamedMarker', 'PublicEvolvingMarker', 'RpcTimeoutMarker', 'BenchmarkMarker', 'NullFromTypeParamMarker', 'UnmodifiableMarker', 'ReferenceMarker', 'SerialMarker', 'ActivateMarker', 'DeactivateMarker', 'JaxrsResourceMarker', 'JSONRequiredMarker', 'WebSocketMarker', 'OnWebSocketCloseMarker', 'OnWebSocketConnectMarker', 'OnWebSocketMessageMarker', 'OnWebSocketErrorMarker', 'ModifiedMarker', 'ExposeMarker', 'PreDestroyMarker', 'EventHandlerMarker', 'NlsSafeMarker', 'NlsMarker', 'ExcludeMarker', 'ShadowMarker', 'TransactionalMarker', 'FinalDefaultMarker', 'ConcurrentMethodMarker', 'OverridingMethodsMustInvokeSuperMarker', 'DialogTitleMarker', 'Log4j2Marker', 'BeanMarker', 'ResourceMarker', 'TooltipMarker', 'DialogMessageMarker', 'ButtonMarker', 'StubbedMarker', 'NotificationTitleMarker', 'ProgressTitleMarker', 'ActionTextMarker', 'InspectionMessageMarker', 'NotificationContentMarker', 'IntentionFamilyNameMarker', 'IntentionNameMarker', 'SafeFieldForPreviewMarker', 'SupportMarker', 'JsNonNullMarker', 'NullMarkedMarker', 'KtPropertyMarker', 'AutoConfigurationMarker', 'JmixPropertyMarker', 'RequestBodyMarker', 'ReadOperationMarker', 'SelectorMarker', 'ParameterizedAdapterTestMarker', 'PathVariableMarker', 'GetExchangeMarker', 'TestValueMarker', 'EnableCachingMarker', 'ParameterizedHttpServerTestMarker', 'PrimaryMarker', 'ConditionalOnMissingBeanMarker', 'ProgrammaticMarker', 'SpringIntegrationTestMarker', 'CreatedDateMarker', 'LastModifiedDateMarker', 'EnableBatchProcessingMarker', 'RepositoryMarker', 'MemberSupportMarker', 'GraphQlExceptionHandlerMarker', 'VolatileMarker', 'CopyMarker', 'InitMarker', 'Other']

def multi_hot_encode(node_types, all_types):
    return [1 if t in node_types else 0 for t in all_types]

def process_ast(ast):
    # Convert AST to PyTorch Geometric format
    node_features = torch.tensor([multi_hot_encode(node['type'], types) for node in ast['nodes']], dtype=torch.float)
    edge_index = []
    for src, dsts in ast['adjacencyList'].items():
        for dst in dsts:
            edge_index.append([int(src), dst])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    labels = torch.tensor([node['nullable'] for node in ast['nodes']], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=labels)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

directory = '/home/ubuntu/spectral_2hv/'
files = glob.glob(os.path.join(directory, '*' + '.json'))

modeldir='/home/ubuntu/nullgcn/'

batch_size=5
epochs=20

# Define the custom order
custom_order = ["data4.json", "data1.json", "data3.json", "data2.json", "data0.json"]
files_sorted = sorted(files, key=lambda x: custom_order.index(os.path.basename(x)))

# Initialize model, optimizer, and criterion
model = GCN(num_node_features=len(types), num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

best_loss = float('inf')

# Training loop
for epoch in range(200):
    total_loss = 0
    total_batches = 0
    for fname in files_sorted:
        cluster = os.path.basename(fname)
        print(f"Processing file: {fname}")
        
        json_data = []
        with open(fname, "r") as file:
            while True:
                obj_str = getobj(file)
                if not obj_str:
                    break
                obj_str = loads(obj_str)
                json_data.append(obj_str)

        # Skip if insufficient data
        if len(json_data) < 2:
            print("Insufficient data")
            continue

        # Process the ASTs and create graphs
        graphs = [process_ast(ast) for ast in json_data]
        data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    epoch_loss = total_loss / total_batches
    print(f'Epoch {epoch}, Loss: {epoch_loss}')

    # Save model if best loss overall, not per cluster
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(modeldir, "best_model.pkl"))
