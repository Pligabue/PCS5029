import fs from "fs"

export const writeResults = () => {
  const dirs = fs.readdirSync("./models")
  const stats = dirs.filter(dir => fs.existsSync(`./models/${dir}/stats.txt`)).map(dir => {
    const fileData = fs.readFileSync(`./models/${dir}/stats.txt`, {encoding: "utf-8"})
    const matches = fileData.match(/Hits: (\d+) \(\d+\.\d+ %\)\nMisses: (\d+) \(\d+\.\d+ %\)/)
    const hyperparams = dir.match(/model_(?<CHAR_ENCODING_SIZE>\d+)_(?<MAX_WORD_SIZE>\d+)_(?<TAG_VECTOR_SIZE>\d+)_(?<RNN_SIZE>\d+)_(?<TIME_STEPS>\d+)_(?<EPOCHS>\d+)_(?<MAX_BATCH>\d+)/).groups
    return {
      model: dir, 
      hits: matches[1],
      misses: matches[2], 
      ratio: Number(matches[1])/(Number(matches[1]) + Number(matches[2])),
      hyperparams: {
        RNN_SIZE: hyperparams.RNN_SIZE,
        TIME_STEPS: hyperparams.TIME_STEPS,
        EPOCHS: hyperparams.EPOCHS
      }
    }
  }).sort((a, b) => Number(b.ratio) - Number(a.ratio))
  
  const statsJSON = JSON.stringify({
    stats: stats,
    createdAt: (new Date()).toISOString()
  }, null, 2)
  
  fs.writeFileSync("stats.json", statsJSON)

  const statsCSV = stats.reduce((acc, {ratio, hyperparams: {RNN_SIZE, TIME_STEPS, EPOCHS}}) => {
    return acc + `${RNN_SIZE},${TIME_STEPS},${EPOCHS},${ratio.toFixed(3)}` + "\n"
  }, "Size of RNN,Time steps,Epochs,Hit ratio\n")

  fs.writeFileSync("stats.csv", statsCSV)
}
