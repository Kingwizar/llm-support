
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build -- --configuration production


FROM node:20-alpine

WORKDIR /app


COPY --from=build /app/dist /app/dist


COPY server.js /app/server.js
COPY package*.json /app/

RUN npm install --omit=dev

EXPOSE 3000
CMD ["node", "server.js"]
