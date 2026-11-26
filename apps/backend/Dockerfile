FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY apps/backend/package.json apps/backend/package-lock.json ./
RUN npm ci
COPY apps/backend/tsconfig.json ./
COPY apps/backend/src/ ./src/
RUN npm run build && npm prune --omit=dev

FROM node:22-bookworm-slim
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules/ ./node_modules/
COPY --from=build /app/dist/ ./dist/
COPY apps/backend/migrations/ ./migrations/
COPY apps/backend/capture/ ./capture/
COPY sample_pool_table1.jpg /sample_pool_table1.jpg
RUN install -d -o node -g node /data
USER node
CMD ["node", "dist/main.js", "api"]
