FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf
COPY talker_server.conf /etc/nginx/conf.d/talker_server.conf
COPY nginx.conf /etc/nginx/nginx.conf