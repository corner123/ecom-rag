ARG MYSQL_BASE_IMAGE=mysql:8.4
FROM ${MYSQL_BASE_IMAGE}
COPY db/init/010_users.sh /docker-entrypoint-initdb.d/010_users.sh
RUN chmod 0555 /docker-entrypoint-initdb.d/010_users.sh
