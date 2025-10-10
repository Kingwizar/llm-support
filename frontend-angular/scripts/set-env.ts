import { writeFileSync } from 'fs';
import { config } from 'dotenv';
import path from 'path';

config({ path: path.resolve(__dirname, '../.env') });

const targetPath = './src/environments/environment.ts';

const chatUrl = `http://127.0.0.1:${process.env.APP_PORT || 8000}/chat`;
const historyUrl = 'http://127.0.0.1:3000/conversations';

const envConfig = `
export const environment = {
  production: ${process.env.APP_ENV === 'production'},
  chatApiUrl: '${chatUrl}',
  historyApiUrl: '${historyUrl}'
};
`;

writeFileSync(targetPath, envConfig);
console.log(`✅ Fichier Angular environment.ts généré depuis .env`);
