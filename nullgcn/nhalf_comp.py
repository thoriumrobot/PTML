import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
import glob
from json_tricks import loads
import random

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
    # Determine the indices of nodes in nodeList
    filtered_node_indices = [i for i, node in enumerate(ast['nodes']) if any(nt in nodeList for nt in node['type'])]

    # Create a mapping from old indices to new indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(filtered_node_indices)}

    # Build edge_index only for edges that connect nodes in nodeList
    edge_index = []
    for src, dsts in ast['adjacencyList'].items():
        if int(src) in filtered_node_indices:
            for dst in dsts:
                if int(dst) in filtered_node_indices:
                    edge_index.append([index_mapping[int(src)], index_mapping[int(dst)]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Prepare node features and labels for the filtered nodes
    node_features = torch.tensor([multi_hot_encode(ast['nodes'][i]['type'], types) for i in filtered_node_indices], dtype=torch.float)
    labels = torch.tensor([ast['nodes'][i]['nullable'] for i in filtered_node_indices], dtype=torch.long)

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

batch_size = 5
epochs = 20

# Define the custom order
custom_order = ["data4.json", "data1.json", "data3.json", "data2.json", "data0.json"]
files_sorted = sorted(files, key=lambda x: custom_order.index(os.path.basename(x)))

# Initialize model
model = GCN(num_node_features=len(types), num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

directory = '/home/ubuntu/spectral_2hv/'
files = glob.glob(os.path.join(directory, '*' + '.json'))
modeldir = '/home/ubuntu/nullgcn/'

batch_size = 20
epochs = 20

# Custom order for processing files
custom_order = ["data4.json", "data1.json", "data3.json", "data2.json", "data0.json"]
files_sorted = sorted(files, key=lambda x: custom_order.index(os.path.basename(x)))

for fname in files_sorted:
    print(f"Processing file: {fname}")
    asts = []  # Reset ASTs for each file

    with open(fname, "r") as file:
        while True:
            obj_str = getobj(file)
            if not obj_str:
                break
            ast = loads(obj_str)
            asts.append(ast)
            
            # Process when we have <batch_size> ASTs or reach the end of the file
            if len(asts) == batch_size or file.tell() == os.fstat(file.fileno()).st_size:
                # Balance dataset by nodes
                nullable_nodes = []
                non_nullable_nodes = []
                for ast in asts:
                    for node in ast['nodes']:
                        if node['nullable'] == 1 and any(nt in nodeList for nt in node['type']):
                            nullable_nodes.append(node)
                        elif node['nullable'] == 0 and any(nt in nodeList for nt in node['type']):
                            non_nullable_nodes.append(node)

                random.shuffle(non_nullable_nodes)
                balanced_nodes = nullable_nodes + non_nullable_nodes[:len(nullable_nodes)]

                best_loss = float('inf')

                # Process nodes in batches
                while balanced_nodes:
                    batch_nodes = balanced_nodes[:min(len(balanced_nodes), batch_size)]
                    balanced_nodes = balanced_nodes[min(len(balanced_nodes), batch_size):]

                    # Create graphs using the full AST, not just the filtered nodes
                    graphs = [process_ast(ast) for ast in asts if any(node in ast['nodes'] for node in balanced_nodes)]

                    if len(graphs)<1:
                        continue

                    data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

                    for epoch in range(epochs):
                        total_loss = 0
                        for data in data_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            out = model(data)
                            loss = criterion(out, data.y)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        epoch_loss = total_loss / len(data_loader)
                        print(f'File {fname}, Epoch {epoch}, Loss: {epoch_loss}')

                        if epoch_loss<best_loss:
                            best_loss=epoch_loss
                            # Save the model after processing each file
                            torch.save(model.state_dict(), os.path.join(modeldir, f"model_after_{os.path.basename(fname)}.pkl"))
            
                # Reset ASTs for the next batch
                asts = []

                # Check if end of file is reached
                if file.tell() == os.fstat(file.fileno()).st_size:
                    break