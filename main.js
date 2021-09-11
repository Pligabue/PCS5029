import tf from "@tensorflow/tfjs-node"
import fs from "fs"

const CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz1234567890".split("")
const CHAR_ENCODING_SIZE = CHARACTER_SET.length + 1
const MAX_WORD_SIZE = 15
const WORD_VECTOR_SIZE = MAX_WORD_SIZE * CHAR_ENCODING_SIZE

const TAGS = ['WH', 'ADV', 'MOD', 'PRON', 'VERB', 'TO', 'DT', 'ADJ', 'NOUN', 'PREP', 'CONJ', 'NUMB', 'PART', 'AUX']
const TAG_VECTOR_SIZE = TAGS.length + 1

const RNN_SIZE = 60
const TIME_STEPS = 20
const EPOCHS = 200

const MAX_BATCH = 1000

const MODEL_PATH = `./models/model_${CHAR_ENCODING_SIZE}_${MAX_WORD_SIZE}_${TAG_VECTOR_SIZE}_${RNN_SIZE}_${TIME_STEPS}_${EPOCHS}_${MAX_BATCH}`

const THRESHOLD = 0.0

const tokenize = (sentence) => {
  return sentence.split(/\s+/)
}

const oneHotEncodeToken = (token) => {
  const characters = token.toLowerCase().split("")
  const characterIndices = characters.map(c => CHARACTER_SET.indexOf(c) + 1)
  characterIndices.length = MAX_WORD_SIZE
  const characterTensor = tf.tensor1d(characterIndices, "int32")
  const encodedCharacters = tf.oneHot(characterTensor, CHAR_ENCODING_SIZE)
  return encodedCharacters.reshape([WORD_VECTOR_SIZE])
}

const encodeSentence = (sentence) => {
  const tokens = tokenize(sentence)
  return tf.stack(tokens.map(oneHotEncodeToken))
}

const padEncodedSentence = (encodedSentence) => {
  const sentenceLength = encodedSentence.shape[0]
  const zeroArr = Array(TIME_STEPS - sentenceLength).fill("")
  const zeros = tf.stack(zeroArr.map(oneHotEncodeToken))
  
  return encodedSentence.concat(zeros)
}

const truncateEncodedSentence = (encodedSentence) => {
  return encodedSentence.slice(0, TIME_STEPS)
}

const buildSentenceInputs = (sentence) => {
  const encodedSentence = encodeSentence(sentence)
  const sentenceLength = encodedSentence.shape[0]
  return sentenceLength < TIME_STEPS ? padEncodedSentence(encodedSentence) : truncateEncodedSentence(encodedSentence) 
}

const oneHotEncodeTag = (tag) => {
  const tag_index = TAGS.indexOf(tag) + 1
  return tf.oneHot(tag_index, TAG_VECTOR_SIZE)
}

const encodeTags = (tags) => {
  return tf.stack(tags.map(oneHotEncodeTag))
}

const padEncodedTags = (encodedTags) => {
  const numberOfTags = encodedTags.shape[0]
  const zeroArr = Array(TIME_STEPS - numberOfTags).fill("")
  const zeros = encodeTags(zeroArr)
  
  return encodedTags.concat(zeros)
}

const truncateEncodedTags = (encodedTags) => {
  return encodedTags.slice(0, TIME_STEPS)
}

const buildTagOutputs = (tags) => {
  const encodedTags = encodeTags(tags)
  const numberOfTags = encodedTags.shape[0]
  return numberOfTags < TIME_STEPS ? padEncodedTags(encodedTags) : truncateEncodedTags(encodedTags) 
}

const decodeOutput = (predictionTensor) => {
  const predictions = predictionTensor.arraySync()
  return predictions.map((prediction) => {
    const maxIndex = prediction.reduce((accIndex, v, i, arr) => (v > arr[accIndex] ? i : accIndex), 1)
    return maxIndex > 0 ? TAGS[maxIndex - 1] : "NO TAG"
  })
}

const data = JSON.parse(fs.readFileSync("data/data.json"))
let model = null

if (fs.existsSync(MODEL_PATH)) {
  model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`)
} else {
  const input = tf.input({shape: [TIME_STEPS, WORD_VECTOR_SIZE]})
  
  const rnn = tf.layers.simpleRNN({units: RNN_SIZE, returnSequences: true});
  const rnn_output = rnn.apply(input);

  const denseLayer = tf.layers.dense({units: TAG_VECTOR_SIZE, activation: "softmax"})
  const output = denseLayer.apply(rnn_output)
  
  model = tf.model({inputs: input, outputs: output})
  model.compile({optimizer: 'adam', loss: tf.losses.cosineDistance});
  
  model.summary()
  
  
  for (let i = 0; i < data.length; i += MAX_BATCH) {
    const batch = data.slice(i, i + MAX_BATCH)
    let xs = [], ys = []
  
    batch.forEach(({sentence, tags}, j) => {
      let sentenceInputs = buildSentenceInputs(sentence)
      let tagOutputs = buildTagOutputs(tags)
  
      if (sentenceInputs.shape[0] === tagOutputs.shape[0]) {
        xs.push(sentenceInputs)
        ys.push(tagOutputs)
      } else {
        console.log(sentence, tags)
      }
    })
  
    await model.fit(tf.stack(xs), tf.stack(ys), {
      epochs: EPOCHS,
      verbose: 0,
      callbacks: {
        onTrainEnd: (logs) => { console.log(`Ended training #${i}-${i + batch.length - 1}`) }
      }
    })
  }

  await model.save(`file://${MODEL_PATH}`)
}

const testSentence = (model, sentence, expectedTags) => {
  console.log(`Testing sentence "${sentence}"`)
  const sentenceInputs = buildSentenceInputs(sentence)
  const outputs = model.predict(tf.stack([sentenceInputs]))
  const outputTags = decodeOutput(tf.unstack(outputs)[0])
  
  const tokens = tokenize(sentence)
  tokens.forEach((token, i) => {
    console.log(token, "=>", outputTags[i], "expected", expectedTags[i])
  })
}

const runTests = (model) => {
  let hits = 0, misses = 0
  data.forEach(({sentence, tags}) => {
    const sentenceInputs = buildSentenceInputs(sentence)
    const outputs = model.predict(tf.stack([sentenceInputs]))

    const tokens = tokenize(sentence)
    const outputTags = decodeOutput(tf.unstack(outputs)[0])

    tokens.forEach((token, i) => {
      if (outputTags[i] === tags[i]) {
        hits++
      } else {
        misses++
      }
    })
  })

  const stats = `Hits: ${hits} (${(100 * hits/(hits + misses)).toFixed(2)} %)\n` +
                `Misses: ${misses} (${(100 * misses/(hits + misses)).toFixed(2)} %)`
  console.log(stats)

  fs.writeFileSync(`${MODEL_PATH}/stats.txt`, stats)
}

runTests(model)

