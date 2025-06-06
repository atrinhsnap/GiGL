// Generated by the Scala Plugin for the Protocol Buffer Compiler.
// Do not edit!
//
// Protofile syntax: PROTO3

package snapchat.research.gbml.inference_metadata

/** @param nodeTypeToInferencerOutputInfoMap
  *   Map of node type to outputs from inferencer
  */
@SerialVersionUID(0L)
final case class InferenceMetadata(
    nodeTypeToInferencerOutputInfoMap: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput] = _root_.scala.collection.immutable.Map.empty,
    unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
    ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[InferenceMetadata] {
    @transient
    private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
    private[this] def __computeSerializedSize(): _root_.scala.Int = {
      var __size = 0
      nodeTypeToInferencerOutputInfoMap.foreach { __item =>
        val __value = snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toBase(__item)
        __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
      }
      __size += unknownFields.serializedSize
      __size
    }
    override def serializedSize: _root_.scala.Int = {
      var __size = __serializedSizeMemoized
      if (__size == 0) {
        __size = __computeSerializedSize() + 1
        __serializedSizeMemoized = __size
      }
      __size - 1
      
    }
    def writeTo(`_output__`: _root_.com.google.protobuf.CodedOutputStream): _root_.scala.Unit = {
      nodeTypeToInferencerOutputInfoMap.foreach { __v =>
        val __m = snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toBase(__v)
        _output__.writeTag(1, 2)
        _output__.writeUInt32NoTag(__m.serializedSize)
        __m.writeTo(_output__)
      };
      unknownFields.writeTo(_output__)
    }
    def clearNodeTypeToInferencerOutputInfoMap = copy(nodeTypeToInferencerOutputInfoMap = _root_.scala.collection.immutable.Map.empty)
    def addNodeTypeToInferencerOutputInfoMap(__vs: (_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput) *): InferenceMetadata = addAllNodeTypeToInferencerOutputInfoMap(__vs)
    def addAllNodeTypeToInferencerOutputInfoMap(__vs: Iterable[(_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput)]): InferenceMetadata = copy(nodeTypeToInferencerOutputInfoMap = nodeTypeToInferencerOutputInfoMap ++ __vs)
    def withNodeTypeToInferencerOutputInfoMap(__v: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput]): InferenceMetadata = copy(nodeTypeToInferencerOutputInfoMap = __v)
    def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
    def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
    def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
      (__fieldNumber: @_root_.scala.unchecked) match {
        case 1 => nodeTypeToInferencerOutputInfoMap.iterator.map(snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toBase(_)).toSeq
      }
    }
    def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
      _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
      (__field.number: @_root_.scala.unchecked) match {
        case 1 => _root_.scalapb.descriptors.PRepeated(nodeTypeToInferencerOutputInfoMap.iterator.map(snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toBase(_).toPMessage).toVector)
      }
    }
    def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
    def companion: snapchat.research.gbml.inference_metadata.InferenceMetadata.type = snapchat.research.gbml.inference_metadata.InferenceMetadata
    // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.InferenceMetadata])
}

object InferenceMetadata extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.inference_metadata.InferenceMetadata] {
  implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.inference_metadata.InferenceMetadata] = this
  def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.inference_metadata.InferenceMetadata = {
    val __nodeTypeToInferencerOutputInfoMap: _root_.scala.collection.mutable.Builder[(_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput), _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput]] = _root_.scala.collection.immutable.Map.newBuilder[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput]
    var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
    var _done__ = false
    while (!_done__) {
      val _tag__ = _input__.readTag()
      _tag__ match {
        case 0 => _done__ = true
        case 10 =>
          __nodeTypeToInferencerOutputInfoMap += snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toCustom(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry](_input__))
        case tag =>
          if (_unknownFields__ == null) {
            _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
          }
          _unknownFields__.parseField(tag, _input__)
      }
    }
    snapchat.research.gbml.inference_metadata.InferenceMetadata(
        nodeTypeToInferencerOutputInfoMap = __nodeTypeToInferencerOutputInfoMap.result(),
        unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
    )
  }
  implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.inference_metadata.InferenceMetadata] = _root_.scalapb.descriptors.Reads{
    case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
      _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
      snapchat.research.gbml.inference_metadata.InferenceMetadata(
        nodeTypeToInferencerOutputInfoMap = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Seq[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry]]).getOrElse(_root_.scala.Seq.empty).iterator.map(snapchat.research.gbml.inference_metadata.InferenceMetadata._typemapper_nodeTypeToInferencerOutputInfoMap.toCustom(_)).toMap
      )
    case _ => throw new RuntimeException("Expected PMessage")
  }
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = InferenceMetadataProto.javaDescriptor.getMessageTypes().get(0)
  def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = InferenceMetadataProto.scalaDescriptor.messages(0)
  def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
    var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
    (__number: @_root_.scala.unchecked) match {
      case 1 => __out = snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry
    }
    __out
  }
  lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] =
    Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]](
      _root_.snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry
    )
  def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
  lazy val defaultInstance = snapchat.research.gbml.inference_metadata.InferenceMetadata(
    nodeTypeToInferencerOutputInfoMap = _root_.scala.collection.immutable.Map.empty
  )
  @SerialVersionUID(0L)
  final case class NodeTypeToInferencerOutputInfoMapEntry(
      key: _root_.scala.Predef.String = "",
      value: _root_.scala.Option[snapchat.research.gbml.inference_metadata.InferenceOutput] = _root_.scala.None,
      unknownFields: _root_.scalapb.UnknownFieldSet = _root_.scalapb.UnknownFieldSet.empty
      ) extends scalapb.GeneratedMessage with scalapb.lenses.Updatable[NodeTypeToInferencerOutputInfoMapEntry] {
      @transient
      private[this] var __serializedSizeMemoized: _root_.scala.Int = 0
      private[this] def __computeSerializedSize(): _root_.scala.Int = {
        var __size = 0
        
        {
          val __value = key
          if (!__value.isEmpty) {
            __size += _root_.com.google.protobuf.CodedOutputStream.computeStringSize(1, __value)
          }
        };
        if (value.isDefined) {
          val __value = value.get
          __size += 1 + _root_.com.google.protobuf.CodedOutputStream.computeUInt32SizeNoTag(__value.serializedSize) + __value.serializedSize
        };
        __size += unknownFields.serializedSize
        __size
      }
      override def serializedSize: _root_.scala.Int = {
        var __size = __serializedSizeMemoized
        if (__size == 0) {
          __size = __computeSerializedSize() + 1
          __serializedSizeMemoized = __size
        }
        __size - 1
        
      }
      def writeTo(`_output__`: _root_.com.google.protobuf.CodedOutputStream): _root_.scala.Unit = {
        {
          val __v = key
          if (!__v.isEmpty) {
            _output__.writeString(1, __v)
          }
        };
        value.foreach { __v =>
          val __m = __v
          _output__.writeTag(2, 2)
          _output__.writeUInt32NoTag(__m.serializedSize)
          __m.writeTo(_output__)
        };
        unknownFields.writeTo(_output__)
      }
      def withKey(__v: _root_.scala.Predef.String): NodeTypeToInferencerOutputInfoMapEntry = copy(key = __v)
      def getValue: snapchat.research.gbml.inference_metadata.InferenceOutput = value.getOrElse(snapchat.research.gbml.inference_metadata.InferenceOutput.defaultInstance)
      def clearValue: NodeTypeToInferencerOutputInfoMapEntry = copy(value = _root_.scala.None)
      def withValue(__v: snapchat.research.gbml.inference_metadata.InferenceOutput): NodeTypeToInferencerOutputInfoMapEntry = copy(value = Option(__v))
      def withUnknownFields(__v: _root_.scalapb.UnknownFieldSet) = copy(unknownFields = __v)
      def discardUnknownFields = copy(unknownFields = _root_.scalapb.UnknownFieldSet.empty)
      def getFieldByNumber(__fieldNumber: _root_.scala.Int): _root_.scala.Any = {
        (__fieldNumber: @_root_.scala.unchecked) match {
          case 1 => {
            val __t = key
            if (__t != "") __t else null
          }
          case 2 => value.orNull
        }
      }
      def getField(__field: _root_.scalapb.descriptors.FieldDescriptor): _root_.scalapb.descriptors.PValue = {
        _root_.scala.Predef.require(__field.containingMessage eq companion.scalaDescriptor)
        (__field.number: @_root_.scala.unchecked) match {
          case 1 => _root_.scalapb.descriptors.PString(key)
          case 2 => value.map(_.toPMessage).getOrElse(_root_.scalapb.descriptors.PEmpty)
        }
      }
      def toProtoString: _root_.scala.Predef.String = _root_.scalapb.TextFormat.printToUnicodeString(this)
      def companion: snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry.type = snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry
      // @@protoc_insertion_point(GeneratedMessage[snapchat.research.gbml.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry])
  }
  
  object NodeTypeToInferencerOutputInfoMapEntry extends scalapb.GeneratedMessageCompanion[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry] {
    implicit def messageCompanion: scalapb.GeneratedMessageCompanion[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry] = this
    def parseFrom(`_input__`: _root_.com.google.protobuf.CodedInputStream): snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry = {
      var __key: _root_.scala.Predef.String = ""
      var __value: _root_.scala.Option[snapchat.research.gbml.inference_metadata.InferenceOutput] = _root_.scala.None
      var `_unknownFields__`: _root_.scalapb.UnknownFieldSet.Builder = null
      var _done__ = false
      while (!_done__) {
        val _tag__ = _input__.readTag()
        _tag__ match {
          case 0 => _done__ = true
          case 10 =>
            __key = _input__.readStringRequireUtf8()
          case 18 =>
            __value = Option(__value.fold(_root_.scalapb.LiteParser.readMessage[snapchat.research.gbml.inference_metadata.InferenceOutput](_input__))(_root_.scalapb.LiteParser.readMessage(_input__, _)))
          case tag =>
            if (_unknownFields__ == null) {
              _unknownFields__ = new _root_.scalapb.UnknownFieldSet.Builder()
            }
            _unknownFields__.parseField(tag, _input__)
        }
      }
      snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry(
          key = __key,
          value = __value,
          unknownFields = if (_unknownFields__ == null) _root_.scalapb.UnknownFieldSet.empty else _unknownFields__.result()
      )
    }
    implicit def messageReads: _root_.scalapb.descriptors.Reads[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry] = _root_.scalapb.descriptors.Reads{
      case _root_.scalapb.descriptors.PMessage(__fieldsMap) =>
        _root_.scala.Predef.require(__fieldsMap.keys.forall(_.containingMessage eq scalaDescriptor), "FieldDescriptor does not match message type.")
        snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry(
          key = __fieldsMap.get(scalaDescriptor.findFieldByNumber(1).get).map(_.as[_root_.scala.Predef.String]).getOrElse(""),
          value = __fieldsMap.get(scalaDescriptor.findFieldByNumber(2).get).flatMap(_.as[_root_.scala.Option[snapchat.research.gbml.inference_metadata.InferenceOutput]])
        )
      case _ => throw new RuntimeException("Expected PMessage")
    }
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.Descriptor = snapchat.research.gbml.inference_metadata.InferenceMetadata.javaDescriptor.getNestedTypes().get(0)
    def scalaDescriptor: _root_.scalapb.descriptors.Descriptor = snapchat.research.gbml.inference_metadata.InferenceMetadata.scalaDescriptor.nestedMessages(0)
    def messageCompanionForFieldNumber(__number: _root_.scala.Int): _root_.scalapb.GeneratedMessageCompanion[_] = {
      var __out: _root_.scalapb.GeneratedMessageCompanion[_] = null
      (__number: @_root_.scala.unchecked) match {
        case 2 => __out = snapchat.research.gbml.inference_metadata.InferenceOutput
      }
      __out
    }
    lazy val nestedMessagesCompanions: Seq[_root_.scalapb.GeneratedMessageCompanion[_ <: _root_.scalapb.GeneratedMessage]] = Seq.empty
    def enumCompanionForFieldNumber(__fieldNumber: _root_.scala.Int): _root_.scalapb.GeneratedEnumCompanion[_] = throw new MatchError(__fieldNumber)
    lazy val defaultInstance = snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry(
      key = "",
      value = _root_.scala.None
    )
    implicit class NodeTypeToInferencerOutputInfoMapEntryLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry](_l) {
      def key: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Predef.String] = field(_.key)((c_, f_) => c_.copy(key = f_))
      def value: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.inference_metadata.InferenceOutput] = field(_.getValue)((c_, f_) => c_.copy(value = Option(f_)))
      def optionalValue: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.Option[snapchat.research.gbml.inference_metadata.InferenceOutput]] = field(_.value)((c_, f_) => c_.copy(value = f_))
    }
    final val KEY_FIELD_NUMBER = 1
    final val VALUE_FIELD_NUMBER = 2
    @transient
    implicit val keyValueMapper: _root_.scalapb.TypeMapper[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry, (_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput)] =
      _root_.scalapb.TypeMapper[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry, (_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput)](__m => (__m.key, __m.getValue))(__p => snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry(__p._1, Some(__p._2)))
    def of(
      key: _root_.scala.Predef.String,
      value: _root_.scala.Option[snapchat.research.gbml.inference_metadata.InferenceOutput]
    ): _root_.snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry = _root_.snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry(
      key,
      value
    )
    // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry])
  }
  
  implicit class InferenceMetadataLens[UpperPB](_l: _root_.scalapb.lenses.Lens[UpperPB, snapchat.research.gbml.inference_metadata.InferenceMetadata]) extends _root_.scalapb.lenses.ObjectLens[UpperPB, snapchat.research.gbml.inference_metadata.InferenceMetadata](_l) {
    def nodeTypeToInferencerOutputInfoMap: _root_.scalapb.lenses.Lens[UpperPB, _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput]] = field(_.nodeTypeToInferencerOutputInfoMap)((c_, f_) => c_.copy(nodeTypeToInferencerOutputInfoMap = f_))
  }
  final val NODE_TYPE_TO_INFERENCER_OUTPUT_INFO_MAP_FIELD_NUMBER = 1
  @transient
  private[inference_metadata] val _typemapper_nodeTypeToInferencerOutputInfoMap: _root_.scalapb.TypeMapper[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry, (_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput)] = implicitly[_root_.scalapb.TypeMapper[snapchat.research.gbml.inference_metadata.InferenceMetadata.NodeTypeToInferencerOutputInfoMapEntry, (_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput)]]
  def of(
    nodeTypeToInferencerOutputInfoMap: _root_.scala.collection.immutable.Map[_root_.scala.Predef.String, snapchat.research.gbml.inference_metadata.InferenceOutput]
  ): _root_.snapchat.research.gbml.inference_metadata.InferenceMetadata = _root_.snapchat.research.gbml.inference_metadata.InferenceMetadata(
    nodeTypeToInferencerOutputInfoMap
  )
  // @@protoc_insertion_point(GeneratedMessageCompanion[snapchat.research.gbml.InferenceMetadata])
}
